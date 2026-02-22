#!/usr/bin/env python
# coding: utf-8
"""
Experiment pipeline for Mixed-Integer Rosenbrock (MIRB).
"""

import time
import numpy as np
import torch
from torch import nn
from tqdm import tqdm

from reins import (
    Variable, TypeVariable, PenaltyLoss, MLPBnDrop,
    LearnableSolver,
)
from reins.variable import VarType
from reins.node import RelaxationNode
from reins.node.rounding import (
    StochasticSTERounding,
    DynamicThresholdRounding,
    StochasticAdaptiveSelectionRounding,
)

from experiments.utils import set_seeds, make_result_df, print_summary, save_results, record_viol, record_failure

import logging
logging.getLogger("pyomo.core").setLevel(logging.ERROR)


def _coefficients(num_blocks):
    """Generate fixed linear constraint coefficients (RandomState(17))."""
    rng = np.random.RandomState(17)
    b = torch.from_numpy(rng.normal(scale=1, size=(num_blocks,))).float()
    q = torch.from_numpy(rng.normal(scale=1, size=(num_blocks,))).float()
    return b, q


def build_loss(x, y, p, a, steepness, num_blocks, penalty_weight, device="cpu", relaxed=False):
    """
    Build PenaltyLoss for the Rosenbrock problem.

    min  sum_i (a_i - x_i)^2 + steepness * (y_i - x_i^2)^2
    s.t. sum(y) >= num_blocks * p / 2      (inner)
         sum(x^2) <= num_blocks * p        (outer)
         b^T x <= 0                        (linear 1)
         q^T y <= 0                        (linear 2)

    Args:
        x: TypeVariable for continuous decision variable.
        y: TypeVariable for integer decision variable.
        p: Variable for the scaling parameter.
        a: Variable for the target parameter.
    """
    x_expr = x.relaxed if relaxed else x
    y_expr = y.relaxed if relaxed else y
    b_coef, q_coef = _coefficients(num_blocks)
    b_coef, q_coef = b_coef.to(device), q_coef.to(device)
    # Objective
    f = torch.sum((a - x_expr) ** 2 + steepness * (y_expr - x_expr ** 2) ** 2, dim=1)
    obj = f.minimize(weight=1.0, name="obj")
    # Constraints
    inner = penalty_weight * (num_blocks * p[:, 0:1] / 2 <= torch.sum(y_expr, dim=1, keepdim=True))
    inner.name = "inner"
    outer = penalty_weight * (torch.sum(x_expr ** 2, dim=1, keepdim=True) <= num_blocks * p[:, 0:1])
    outer.name = "outer"
    linear1 = penalty_weight * ((x_expr @ b_coef) <= 0)
    linear1.name = "linear1"
    linear2 = penalty_weight * ((y_expr @ q_coef) <= 0)
    linear2.name = "linear2"
    return PenaltyLoss(objectives=[obj], constraints=[inner, outer, linear1, linear2])


def run_EX(loader_test, config):
    # Set random seeds for reproducibility
    set_seeds()
    # Print experiment info
    print(config)
    print(f"EX in RB for size {config.size}.")
    steepness = config.steepness
    num_blocks = config.size
    # Init exact solver
    from experiments.math_solver.rosenbrock import rosenbrock
    model = rosenbrock(steepness, num_blocks, timelimit=1000)
    # Init result lists
    params, sols, viols, objvals, mean_viols, max_viols, num_viols, elapseds = [], [], [], [], [], [], [], []
    # Evaluate on test set
    p_test_np = torch.as_tensor(loader_test.dataset.datadict["p"][:100]).cpu().numpy()
    a_test_np = torch.as_tensor(loader_test.dataset.datadict["a"][:100]).cpu().numpy()
    # Loop over test samples
    for i in tqdm(range(100)):
        p, a = p_test_np[i], a_test_np[i]
        # Set parameter values
        model.set_param_val({"p": p, "a": a})
        tick = time.time()
        params.append(p.tolist() + a.tolist())
        # Solve and record results
        try:
            xval, objval = model.solve()
            tock = time.time()
            sols.append(list(xval["x"].values()) + list(xval["y"].values()))
            objvals.append(objval)
            record_viol(model, viols, mean_viols, max_viols, num_viols)
        except Exception:
            record_failure(sols, viols, objvals, mean_viols, max_viols, num_viols)
            tock = time.time()
        # Record elapsed time
        elapseds.append(tock - tick)
    # Create result dataframe and print summary
    df = make_result_df(params, sols, viols, objvals, mean_viols, max_viols, num_viols, elapseds)
    print_summary(df, show_unsolved=True, sleep=True)
    # Save .npz (sol arrays) and .csv (statistics)
    save_results(df,
                 f"result/sol/rb_exact_{num_blocks}.npz",
                 f"result/stat/rb_exact_{num_blocks}.csv")


def run_RR(loader_test, config):
    # Set random seeds for reproducibility
    set_seeds()
    # Print experiment info
    print(config)
    print(f"RR in RB for size {config.size}.")
    from experiments.heuristic import naive_round
    steepness = config.steepness
    num_blocks = config.size
    # Init relaxed solver
    from experiments.math_solver.rosenbrock import rosenbrock
    model = rosenbrock(steepness, num_blocks, timelimit=1000)
    # Init result lists
    params, sols, viols, objvals, mean_viols, max_viols, num_viols, elapseds = [], [], [], [], [], [], [], []
    # Evaluate on test set
    p_test_np = torch.as_tensor(loader_test.dataset.datadict["p"][:100]).cpu().numpy()
    a_test_np = torch.as_tensor(loader_test.dataset.datadict["a"][:100]).cpu().numpy()
    for i in tqdm(range(100)):
        p, a = p_test_np[i], a_test_np[i]
        # Set parameter values and relax integrality
        model.set_param_val({"p": p, "a": a})
        model_rel = model.relax()
        tick = time.time()
        params.append(p.tolist() + a.tolist())
        # Solve relaxation then naive-round to integers
        try:
            xval_rel, _ = model_rel.solve()
            xval, objval = naive_round(xval_rel, model)
            tock = time.time()
            sols.append(list(xval["x"].values()) + list(xval["y"].values()))
            objvals.append(objval)
            record_viol(model, viols, mean_viols, max_viols, num_viols)
        except Exception:
            record_failure(sols, viols, objvals, mean_viols, max_viols, num_viols)
            tock = time.time()
        # Record elapsed time
        elapseds.append(tock - tick)
    # Create result dataframe and print summary
    df = make_result_df(params, sols, viols, objvals, mean_viols, max_viols, num_viols, elapseds)
    print_summary(df, show_unsolved=True, sleep=True)
    # Save .npz (sol arrays) and .csv (statistics)
    save_results(df,
                 f"result/sol/rb_rel_{num_blocks}.npz",
                 f"result/stat/rb_rel_{num_blocks}.csv")


def run_N1(loader_test, config):
    # Set random seeds for reproducibility
    set_seeds()
    # Print experiment info
    print(config)
    print(f"N1 in RB for size {config.size}.")
    steepness = config.steepness
    num_blocks = config.size
    # Init heuristic solver (1-node B&B)
    from experiments.math_solver.rosenbrock import rosenbrock
    model = rosenbrock(steepness, num_blocks, timelimit=1000)
    model_heur = model.first_solution_heuristic(nodes_limit=1)
    # Init result lists
    params, sols, viols, objvals, mean_viols, max_viols, num_viols, elapseds = [], [], [], [], [], [], [], []
    # Evaluate on test set
    p_test_np = torch.as_tensor(loader_test.dataset.datadict["p"][:100]).cpu().numpy()
    a_test_np = torch.as_tensor(loader_test.dataset.datadict["a"][:100]).cpu().numpy()
    for i in tqdm(range(100)):
        p, a = p_test_np[i], a_test_np[i]
        # Set parameter values
        model_heur.set_param_val({"p": p, "a": a})
        tick = time.time()
        params.append(p.tolist() + a.tolist())
        # Solve and record results
        try:
            xval, objval = model_heur.solve()
            tock = time.time()
            sols.append(list(xval["x"].values()) + list(xval["y"].values()))
            objvals.append(objval)
            record_viol(model_heur, viols, mean_viols, max_viols, num_viols)
        except Exception:
            record_failure(sols, viols, objvals, mean_viols, max_viols, num_viols)
            tock = time.time()
        # Record elapsed time
        elapseds.append(tock - tick)
    # Create result dataframe and print summary
    df = make_result_df(params, sols, viols, objvals, mean_viols, max_viols, num_viols, elapseds)
    print_summary(df, show_unsolved=True, sleep=True)
    # Save .npz (sol arrays) and .csv (statistics)
    save_results(df,
                 f"result/sol/rb_root_{num_blocks}.npz",
                 f"result/stat/rb_root_{num_blocks}.csv")


def _run_network_rounding(loader_train, loader_test, loader_val, config,
                           rounding_cls, prefix, label):
    """
    Shared logic for AS and DT (network-based) rounding methods.
    run_AS and run_DT differ only in rounding_cls and file prefix,
    so they are unified here to avoid duplication.
    """
    # Set random seeds for reproducibility
    set_seeds()
    # Print experiment info
    print(config)
    print(f"{label} in RB for size {config.size}.")
    steepness = config.steepness
    num_blocks = config.size
    hsize, hlayers_sol, hlayers_rnd = config.hsize, config.hlayers_sol, config.hlayers_rnd
    lr, penalty_weight = config.lr, config.penalty
    # Build loss and get typed variables
    x = TypeVariable("x", num_vars=num_blocks)
    y = TypeVariable("y", num_vars=num_blocks, var_types=VarType.INTEGER)
    p = Variable("p")
    a = Variable("a")
    loss = build_loss(x, y, p, a, steepness, num_blocks, penalty_weight, device="cuda")
    # Create solution mapping network
    rel_func = MLPBnDrop(insize=num_blocks + 1, outsize=2 * num_blocks,
                          hsizes=[hsize] * hlayers_sol, nonlin=nn.ReLU)
    rel = RelaxationNode(rel_func, [p, a], [x, y], name="relaxation")
    # Create rounding network and operator
    rnd_net = MLPBnDrop(insize=3 * num_blocks + 1, outsize=2 * num_blocks,
                        hsizes=[hsize] * hlayers_rnd)
    rnd = rounding_cls(rnd_net, [p, a], [x, y], continuous_update=True)
    # Set up solver
    proj_steps = 10000 if config.project else 0
    solver = LearnableSolver(rel, rnd, loss, projection_steps=proj_steps)
    # Set up optimizer
    optimizer = torch.optim.AdamW(solver.problem.parameters(), lr=lr)
    # Train
    solver.train(loader_train, loader_val, optimizer, device="cuda")
    # Evaluate on test set
    from experiments.math_solver.rosenbrock import rosenbrock
    model = rosenbrock(steepness, num_blocks, timelimit=1000)
    df = evaluate(solver, model, loader_test)
    # Save results
    suffix = "-p" if config.project else ""
    save_results(df,
                 f"result/sol/{prefix}{penalty_weight}_{num_blocks}{suffix}.npz",
                 f"result/stat/{prefix}{penalty_weight}_{num_blocks}{suffix}.csv")


def run_AS(loader_train, loader_test, loader_val, config):
    """Adaptive selection rounding (Gumbel)."""
    _run_network_rounding(loader_train, loader_test, loader_val, config,
                          StochasticAdaptiveSelectionRounding, "rb_cls", "AS")


def run_DT(loader_train, loader_test, loader_val, config):
    """Dynamic threshold rounding."""
    _run_network_rounding(loader_train, loader_test, loader_val, config,
                          DynamicThresholdRounding, "rb_thd", "DT")


def run_RS(loader_train, loader_test, loader_val, config):
    """STE rounding."""
    # Set random seeds for reproducibility
    set_seeds()
    # Print experiment info
    print(config)
    print(f"STE in RB for size {config.size}.")
    steepness = config.steepness
    num_blocks = config.size
    hsize, hlayers_sol = config.hsize, config.hlayers_sol
    lr, penalty_weight = config.lr, config.penalty
    # Build loss and get typed variables
    x = TypeVariable("x", num_vars=num_blocks)
    y = TypeVariable("y", num_vars=num_blocks, var_types=VarType.INTEGER)
    p = Variable("p")
    a = Variable("a")
    loss = build_loss(x, y, p, a, steepness, num_blocks, penalty_weight, device="cuda")
    # Create solution mapping network
    rel_func = MLPBnDrop(insize=num_blocks + 1, outsize=2 * num_blocks,
                          hsizes=[hsize] * hlayers_sol, nonlin=nn.ReLU)
    rel = RelaxationNode(rel_func, [p, a], [x, y], name="relaxation")
    # Create rounding operator (STE: no additional network needed)
    rnd = StochasticSTERounding([x, y])
    # Set up solver
    proj_steps = 10000 if config.project else 0
    solver = LearnableSolver(rel, rnd, loss, projection_steps=proj_steps)
    # Set up optimizer
    optimizer = torch.optim.AdamW(solver.problem.parameters(), lr=lr)
    # Train
    solver.train(loader_train, loader_val, optimizer, device="cuda")
    # Evaluate on test set
    from experiments.math_solver.rosenbrock import rosenbrock
    model = rosenbrock(steepness, num_blocks, timelimit=1000)
    df = evaluate(solver, model, loader_test)
    # Save results
    suffix = "-p" if config.project else ""
    save_results(df,
                 f"result/sol/rb_ste{penalty_weight}_{num_blocks}{suffix}.npz",
                 f"result/stat/rb_ste{penalty_weight}_{num_blocks}{suffix}.csv")


def run_LR(loader_train, loader_test, loader_val, config):
    """Learn-then-round: train solution map without rounding, naive round at test."""
    # Set random seeds for reproducibility
    set_seeds()
    # Print experiment info
    print(config)
    print(f"LR in RB for size {config.size}.")
    from reins import Problem, Trainer
    steepness = config.steepness
    num_blocks = config.size
    hsize, hlayers_sol = config.hsize, config.hlayers_sol
    lr, penalty_weight = config.lr, config.penalty
    # Build loss (relaxed: loss reads x_rel/y_rel directly, no rounding layer)
    x = TypeVariable("x", num_vars=num_blocks)
    y = TypeVariable("y", num_vars=num_blocks, var_types=VarType.INTEGER)
    p = Variable("p")
    a = Variable("a")
    loss = build_loss(x, y, p, a, steepness, num_blocks, penalty_weight, device="cuda", relaxed=True)
    # Create solution mapping network (no rounding layer)
    rel_func = MLPBnDrop(insize=num_blocks + 1, outsize=2 * num_blocks,
                          hsizes=[hsize] * hlayers_sol, nonlin=nn.ReLU)
    rel = RelaxationNode(rel_func, [p, a], [x, y], name="relaxation")
    # Set up problem and train
    problem = Problem(nodes=[rel], loss=loss)
    problem.to("cuda")
    optimizer = torch.optim.AdamW(problem.parameters(), lr=lr)
    trainer = Trainer(problem, loader_train, loader_val, optimizer=optimizer, device="cuda")
    best_model = trainer.train()
    problem.load_state_dict(best_model)
    # Evaluate with naive rounding
    from experiments.heuristic import naive_round
    from experiments.math_solver.rosenbrock import rosenbrock
    model = rosenbrock(steepness, num_blocks, timelimit=1000)
    params, sols, viols, objvals, mean_viols, max_viols, num_viols, elapseds = [], [], [], [], [], [], [], []
    # Batch inference for the entire test slice
    p_test_all = torch.as_tensor(loader_test.dataset.datadict["p"][:100]).to("cuda")
    a_test_all = torch.as_tensor(loader_test.dataset.datadict["a"][:100]).to("cuda")
    problem.eval()
    tick_inf = time.time()
    with torch.no_grad():
        test_results = rel({"p": p_test_all, "a": a_test_all})
    tock_inf = time.time()
    # Convert results to numpy for post-processing
    x_all_np = test_results["x_rel"].detach().cpu().numpy()
    y_all_np = test_results["y_rel"].detach().cpu().numpy()
    p_all_np = p_test_all.detach().cpu().numpy()
    a_all_np = a_test_all.detach().cpu().numpy()
    inf_time_per_sample = (tock_inf - tick_inf) / 100
    # Post-process each sample: set model values, naive-round, record results
    for i in tqdm(range(100)):
        p_np, a_np = p_all_np[i], a_all_np[i]
        model.set_param_val({"p": p_np, "a": a_np})
        for j, val in enumerate(x_all_np[i]):
            model.vars["x"][j].value = val
        for j, val in enumerate(y_all_np[i]):
            model.vars["y"][j].value = val
        xval_rel, _ = model.get_val()
        xval, objval = naive_round(xval_rel, model)
        params.append(p_np.tolist() + a_np.tolist())
        sols.append(list(xval["x"].values()) + list(xval["y"].values()))
        objvals.append(objval)
        record_viol(model, viols, mean_viols, max_viols, num_viols)
        elapseds.append(inf_time_per_sample)
    # Create result dataframe and print summary
    df = make_result_df(params, sols, viols, objvals, mean_viols, max_viols, num_viols, elapseds)
    print_summary(df, sleep=True)
    # Save .npz (sol arrays) and .csv (statistics)
    save_results(df,
                 f"result/sol/rb_lrn{penalty_weight}_{num_blocks}.npz",
                 f"result/stat/rb_lrn{penalty_weight}_{num_blocks}.csv")


def evaluate(solver, model, loader_test):
    """Evaluate a LearnableSolver against the math solver ground truth."""
    # Initialize result lists
    params, sols, viols, objvals, mean_viols, max_viols, num_viols, elapseds = [], [], [], [], [], [], [], []
    # Batch inference for the entire test slice
    p_test_all = torch.as_tensor(loader_test.dataset.datadict["p"][:100]).to("cuda")
    a_test_all = torch.as_tensor(loader_test.dataset.datadict["a"][:100]).to("cuda")
    tick_inf = time.time()
    test_results = solver.predict({"p": p_test_all, "a": a_test_all})
    tock_inf = time.time()
    # Get per-sample projection iteration counts
    proj_iters = test_results["_proj_iters"].tolist()
    # Convert results to numpy for post-processing
    x_all_np = test_results["x"].detach().cpu().numpy()
    y_all_np = test_results["y"].detach().cpu().numpy()
    p_all_np = p_test_all.detach().cpu().numpy()
    a_all_np = a_test_all.detach().cpu().numpy()
    inf_time_per_sample = (tock_inf - tick_inf) / 100
    # Post-process each sample: set model values, get solution, record results
    for i in tqdm(range(100)):
        p_np, a_np = p_all_np[i], a_all_np[i]
        model.set_param_val({"p": p_np, "a": a_np})
        for j, val in enumerate(x_all_np[i]):
            model.vars["x"][j].value = val
        for j, val in enumerate(y_all_np[i]):
            model.vars["y"][j].value = val
        # Get solution and objective value
        xval, objval = model.get_val()
        # Record results
        params.append(p_np.tolist() + a_np.tolist())
        sols.append(list(xval["x"].values()) + list(xval["y"].values()))
        objvals.append(objval)
        record_viol(model, viols, mean_viols, max_viols, num_viols)
        elapseds.append(inf_time_per_sample)
    # Create result dataframe and print summary
    df = make_result_df(params, sols, viols, objvals, mean_viols, max_viols, num_viols, elapseds,
                        proj_iters=proj_iters)
    print_summary(df)
    return df
