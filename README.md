# REINS: Relaxation-Enforced Integer Neural Neighbourhood Search for Parametric MINLP with Feasibility Guarantees

![Framework](img/pipeline.png)

Based on the paper: **"[Learning to Optimize for Mixed-Integer Nonlinear Programming with Feasibility Guarantees](https://arxiv.org/abs/2410.11061)"**

## Overview

REINS solves **parametric MINLP**: given a family of optimization problems that share the same structure but differ in parameter values (e.g., constraint right-hand sides), it learns a neural network that maps parameters directly to high-quality integer solutions, without invoking a traditional solver at inference time.

The key components are differentiable **integer correction layers** (for rounding continuous relaxations to integers) and **gradient-based feasibility projection** (for enforcing feasibility post-hoc). Training is self-supervised: only sampled parameter values are needed, not expensive optimal solutions. The framework scales to problems with tens of thousands of variables at subsecond inference with fast training.

## Citation

```bibtex
@article{tang2024learning,
  title={Learning to optimize for mixed-integer non-linear programming with feasibility guarantees},
  author={Tang, Bo and Khalil, Elias B and Drgo{\v{n}}a, J{\'a}n},
  journal={arXiv preprint arXiv:2410.11061},
  year={2024}
}
```

## Slides

Our talk at ICS 2025. View the slides [here](https://github.com/pnnl/L2O-pMINLP/blob/master/slides/L2O-MINLP.pdf).


## Installation

```bash
pip install -e .
```

**Requirements:** Python >= 3.10, PyTorch, NeuroMANCER


## Tutorial

This tutorial walks through building a learnable solver for a parametric integer quadratic program:

$$\min_{x} \quad \frac{1}{2} x^\top Q x + c^\top x \quad \text{s.t.} \quad Ax \leq b, \quad x \in \mathbb{Z}^n$$

Here $Q$, $c$, $A$ are fixed problem coefficients, while $b$ is the **varying parameter**. Different values of $b$ define different problem instances. The goal of REINS is to learn a neural network mapping $b \mapsto x^*$ so that, given any new $b$, the network efficiently predicts a high-quality integer solution.

The training pipeline:
1. **Sample** a dataset of parameter values $\{b^{(i)}\}$ (no optimal solutions needed)
2. **Train** the network end-to-end with a self-supervised penalty loss (objective + constraint violations)
3. **Predict** solutions for unseen parameters via a single forward pass


### Step 1: Define Variables

Use `TypeVariable` for **decision variables** with type metadata (tells rounding layers which indices need integrality enforcement), and `Variable` for **parameters** (continuous inputs like constraint RHS).

```python
from reins import TypeVariable, Variable, VarType

# Pure integer decision variable
x = TypeVariable("x", num_vars=5, var_types=VarType.INTEGER)

# Equivalent using index-based specification
x = TypeVariable("x", num_vars=5, integer_indices=[0, 1, 2, 3, 4])

# Mixed-integer: indices 0,1 are integer, index 2 is binary, rest continuous
y = TypeVariable("y", num_vars=5, integer_indices=[0, 1], binary_indices=[2])
# Equivalent using explicit type list
y = TypeVariable("y", var_types=[
    VarType.INTEGER, VarType.INTEGER, VarType.BINARY,
    VarType.CONTINUOUS, VarType.CONTINUOUS,
])

# Parameter variable (continuous, no type metadata)
b = Variable("b")
```


### Step 2: Define Loss (Objectives + Constraints)

Define objectives and constraints symbolically via operator overloading, then combine into a `PenaltyLoss`. **Use the same `x` and `b` from Step 1** so that the loss and rounding layer share the same variable objects.

```python
import torch
import numpy as np
from reins import PenaltyLoss

# Fixed problem coefficients
rng = np.random.RandomState(17)
Q = torch.from_numpy(0.01 * np.diag(rng.random(size=num_var))).float()
c = torch.from_numpy(0.1 * rng.random(num_var)).float()
A = torch.from_numpy(rng.normal(scale=0.1, size=(num_ineq, num_var))).float()

# Objective: minimize (1/2) x^T Q x + c^T x
f = 0.5 * torch.sum((x @ Q) * x, dim=1) + torch.sum(c * x, dim=1)
obj = f.minimize(weight=1.0, name="obj")

# Constraint: Ax <= b
penalty_weight = 100
con = penalty_weight * (x @ A.T <= b)

loss = PenaltyLoss(objectives=[obj], constraints=[con])
```


### Step 3: Build Relaxation Network

The relaxation network learns the mapping $b \mapsto x_{\text{rel}}$. Wrap any PyTorch module in a `RelaxationNode` to integrate it into the pipeline.

```python
from reins import MLPBnDrop
from reins.node import RelaxationNode

num_var = 5
num_ineq = 5

rel_net = MLPBnDrop(
    insize=num_ineq,
    outsize=num_var,
    hsizes=[64] * 4,
    dropout=0.2,      # dropout rate
    bnorm=True,       # batch normalization
)

# data["b"] -> rel_net -> data["x_rel"]
rel = RelaxationNode(rel_net, [b], [x])
```


### Step 4: Choose a Rounding Layer

Rounding layers convert continuous relaxations to integer solutions.

```python
from reins.node.rounding import (
    StochasticAdaptiveSelectionRounding,
    DynamicThresholdRounding,
)

rnd_net = MLPBnDrop(
    insize=num_ineq + num_var,
    outsize=num_var,
    hsizes=[64] * 3,
)

# Adaptive Selection (AS)
rounding = StochasticAdaptiveSelectionRounding(
    rnd_net, [b], [x], continuous_update=True,
)

# Dynamic Thresholding (DT)
rounding = DynamicThresholdRounding(
    rnd_net, [b], [x],
)
```


### Step 5: Assemble the Solver

`LearnableSolver` composes the relaxation node, rounding layer, and loss.

```python
from reins import LearnableSolver

# Default: GradientProjection enabled (1000 steps) for feasibility enforcement at inference
solver = LearnableSolver(rel, rounding, loss)

# Disable projection
solver = LearnableSolver(rel, rounding, loss, projection_steps=0)
```


### Step 6: Prepare Data & Train

Training data consists of sampled parameter values only — no optimal solutions needed.

```python
from torch.utils.data import DataLoader
from reins import DictDataset

num_data = 10000
b_samples = torch.from_numpy(
    np.random.uniform(-1, 1, size=(num_data, num_ineq))
).float()

data_train = DictDataset({"b": b_samples[:8000]}, name="train")
data_val   = DictDataset({"b": b_samples[8000:9000]}, name="val")
data_test  = DictDataset({"b": b_samples[9000:]}, name="test")

loader_train = DataLoader(data_train, batch_size=64, shuffle=True,
                          collate_fn=data_train.collate_fn)
loader_val   = DataLoader(data_val, batch_size=64, shuffle=False,
                          collate_fn=data_val.collate_fn)

optimizer = torch.optim.AdamW(solver.problem.parameters(), lr=1e-3)
solver.train(
    loader_train,
    loader_val,
    optimizer,
    epochs=200,      # max epochs
    patience=20,     # early stopping patience
    warmup=20,       # warmup epochs before early stopping kicks in
    device="cuda",
)
```


### Step 7: Predict

```python
b_test = data_test.datadict["b"].to("cuda")
result = solver.predict({"b": b_test, "name": "test"})

print(result["x"])       # integer solution
print(result["x_rel"])   # continuous relaxation
```

## Package Structure

```
src/reins/                    # Core package
├── __init__.py                  # Public API
├── variable.py                  # VarType enum & TypeVariable class
├── blocks.py                    # MLPBnDrop (MLP with BatchNorm + Dropout)
├── solver.py                    # LearnableSolver wrapper
├── node/                        # Node components
│   ├── relaxation.py            # RelaxationNode (relaxation solution)
│   └── rounding/                # Integer rounding layers
│       ├── functions.py         # Differentiable STE primitives
│       ├── base.py              # RoundingNode abstract base class
│       ├── ste.py               # STERounding, StochasticSTERounding
│       ├── threshold.py         # DynamicThresholdRounding, StochasticDynamicThresholdRounding
│       └── selection.py         # AdaptiveSelectionRounding, StochasticAdaptiveSelectionRounding
├── projection/                  # Feasibility projection
│   └── gradient.py              # GradientProjection
└── utils/
experiments/                     # Benchmark experiments (not part of the package)
├── quadratic.py                 # Integer Quadratic Programming (IQP)
├── nonconvex.py                 # Integer Non-Convex Programming (INP)
├── rosenbrock.py                # Mixed-Integer Rosenbrock (MIRB)
├── heuristic.py                 # Rounding heuristics (naive_round, floor_round)
└── math_solver/                 # Pyomo-based exact solvers for evaluation
    ├── abc_solver.py            # Abstract parametric solver base class
    ├── quadratic.py             # IQP solver (Gurobi)
    ├── nonconvex.py             # INP solver (SCIP)
    └── rosenbrock.py            # MIRB solver (SCIP)
tests/                           # pytest test suite
```

## Methodology

### Integer Correction Layers

Two learnable correction layers enforce integrality in neural network output:

<div align="center">
    <img src="img/method_RC.png" alt="example for RC" width="48%"/>
    <img src="img/method_LT.png" alt="example for RT" width="48%"/>
</div>

- **Adaptive Selection (AS)** / `AdaptiveSelectionRounding`: Learns a classification strategy to determine rounding directions for integer variables.
- **Dynamic Thresholding (DT)** / `DynamicThresholdRounding`: Learns a threshold value for each integer variable to decide whether to round up or down.

### Integer Feasibility Projection

A gradient-based projection iteratively refines infeasible solutions. The figure below illustrates how the projection step adjusts a solution over multiple iterations.

<div align="center"> <img src="img/example2.png" alt="Feasibility Projection Iterations" width="40%"/> </div>

By integrating feasibility projection with the correction layers, we extend AS and DT into **AS-P** and **DT-P**, respectively.


## Performance

Our learning-based methods (AS & DT) achieve comparable or superior performance to exact solvers (EX) while being orders of magnitude faster:

<div align="center">
    <img src="img/cq_s100_penalty.png" alt="Penalty Effect on IQP" width="40%"/>
    <img src="img/rb_s100_penalty.png" alt="Penalty Effect on MIRB" width="40%"/>
</div>

With properly tuned penalty weights, the approach attains comparable or better objective values within sub-seconds, while exact solvers require up to 1000 seconds.


## Reproducibility

Three benchmark problems are included:

- **Integer Quadratic Problems (IQP)**: Convex quadratic objective with linear constraints and integer variables.
- **Integer Non-Convex Problems (INP)**: Non-convex variant with trigonometric terms.
- **Mixed-Integer Rosenbrock Problems (MIRB)**: Rosenbrock function with linear and non-linear constraints.

```bash
python run_qp.py --size 5
python run_nc.py --size 10 --penalty 1 --project
python run_rb.py --size 100 --penalty 10 --project
```

Arguments: `--size` (problem size), `--penalty` (constraint violation weight), `--project` (enable feasibility projection).


## License

MIT
