"""
Parametric Mixed Integer Nonlinear Programming with SCIP
"""

from abc import ABC, abstractmethod
import copy
from pathlib import Path

import numpy as np
from pyomo import environ as pe
from pyomo import opt as po
from pyomo.core import TransformationFactory

class abcParamSolver(ABC):
    @abstractmethod
    def __init__(self, solver="scip", timelimit=None):
        # Create solver instance
        self.solver = solver
        self.opt = po.SolverFactory(solver)
        # Set time limit
        if timelimit:
            if self.solver == "scip":
                self.opt.options["limits/time"] = timelimit
            elif self.solver == "gurobi":
                self.opt.options["timelimit"] = timelimit
            else:
                raise ValueError("Solver '{}' does not support setting a time limit.".format(solver))
        # Initialize attributes
        self.model = None
        self.params = {}
        self.vars = {}
        self.cons = None
        self._has_warm_start = False

    @property
    def int_ind(self):
        """Indices of integer variables per variable group."""
        int_ind = {}
        for key, var_comp in self.vars.items():
            int_ind[key] = [i for i, v in var_comp.items() if v.domain is pe.Integers]
        return int_ind

    @property
    def bin_ind(self):
        """Indices of binary variables per variable group."""
        bin_ind = {}
        for key, var_comp in self.vars.items():
            bin_ind[key] = [i for i, v in var_comp.items() if v.domain is pe.Binary]
        return bin_ind

    def solve(self, tee=False, keepfiles=False, logfile=None):
        """Solve the model and return variable values and objective value."""
        # Check logfile directory
        if logfile:
            Path(logfile).parent.mkdir(parents=True, exist_ok=True)
        # Clear variable values
        if not self._has_warm_start:
            for var in self.model.component_objects(pe.Var, active=True):
                for index in var:
                    var[index].value = None
        # Solve the model
        if self.solver in ["gurobi", "cplex", "xpress"]:
            self.res = self.opt.solve(self.model, warmstart=self._has_warm_start,
                                      tee=tee, keepfiles=keepfiles, logfile=logfile)
        else:
            self.res = self.opt.solve(self.model, tee=tee, keepfiles=keepfiles,
                                      logfile=logfile)
        # Reset warm start flag
        self._has_warm_start = False
        # Get variable values and objective value
        xval, objval = self.get_val()
        return xval, objval

    def set_param_val(self, param_dict):
        """Set values for mutable parameters."""
        # Iterate through parameter categories
        for key, val in param_dict.items():
            param = self.params[key]
            if isinstance(val, (np.ndarray, list)) and param.is_indexed():
                # Pyomo bulk update for indexed params
                param.store_values(dict(enumerate(val)))
            # Set single value (scalar param)
            else:
                param.set_value(val)
        # Reset warm start flag
        self._has_warm_start = False

    def get_val(self):
        """Retrieve decision variable values and objective value."""
        # Get variable values as dict
        solvals = {}
        try:
            for key, var_comp in self.vars.items():
                vals = {i: float(var_comp[i].value) for i in var_comp}
                if any(np.isnan(v) for v in vals.values()):
                    return None, None
                solvals[key] = vals
            # Get the objective value
            objval = float(pe.value(self.model.obj))
            if np.isnan(objval):
                return None, None
        except (ValueError, AttributeError, TypeError):
            # No value or invalid state
            return None, None
        return solvals, objval

    def set_warm_start(self, init_sol):
        """Set an initial solution for warm starting."""
        for key, vals in init_sol.items():
            if key not in self.vars:
                raise KeyError(f"Variable group '{key}' not found in self.vars")
            var_comp = self.vars[key]
            for i, v in vals.items():
                if i not in var_comp:
                    raise KeyError(f"Index '{i}' not in variable group '{key}'")
                # Assign warm start value
                var_comp[i].set_value(v)
                var_comp[i].stale = False
        # Set warm start flag
        self._has_warm_start = True

    def check_violation(self):
        """Check for any constraint violations."""
        return any(self._constraint_violation(constr) != 0 for constr in self.model.cons.values())

    def cal_violation(self):
        """Calculate violation magnitude for each constraint."""
        return np.array([self._constraint_violation(constr) for constr in self.model.cons.values()])

    def _constraint_violation(self, constr):
        """Compute the violation of a single constraint."""
        lhs = float(pe.value(constr.body))
        if np.isnan(lhs):
            return float("inf")
        # Check if LHS is below the lower bound
        if constr.lower is not None:
            lb = float(pe.value(constr.lower))
            if lhs < lb - 1e-5:
                return lb - lhs
        # Check if LHS is above the upper bound
        if constr.upper is not None:
            ub = float(pe.value(constr.upper))
            if lhs > ub + 1e-5:
                return lhs - ub
        return 0.0

    def clone(self):
        """Return a deep copy of the solver."""
        # Shallow copy the solver
        model_new = copy.copy(self)
        # Clone Pyomo model
        model_new.model = self.model.clone()
        # Deep copy solver instance for independent options
        model_new.opt = copy.deepcopy(self.opt)
        # Rebind variables
        model_new.vars = {var: getattr(model_new.model, var) for var in self.vars}
        # Rebind constraints
        model_new.cons = model_new.model.cons
        # Rebind parameters
        model_new.params = {param: getattr(model_new.model, param) for param in self.params}
        return model_new

    def relax(self):
        """Return a copy with integer/binary variables relaxed to continuous."""
        # Clone model
        model_rel = self.clone()
        # Relax integer variables to continuous
        TransformationFactory("core.relax_integer_vars").apply_to(model_rel.model)
        # Set iteration limits
        if self.solver == "scip":
            model_rel.opt.options["limits/totalnodes"] = 100
            model_rel.opt.options["lp/iterlim"] = 100
        elif self.solver == "gurobi":
            model_rel.opt.options["NodeLimit"] = 100
        else:
            raise ValueError("Solver '{}' does not support setting a total nodes limit.".format(self.solver))
        return model_rel

    def penalty(self, weight):
        """Return a copy with constraints converted to soft penalties."""
        # Clone model
        model_pen = self.clone()
        model = model_pen.model
        # Slack variables
        model.slack = pe.Var(pe.Set(initialize=model.cons.keys()), domain=pe.NonNegativeReals)
        # Add slacks to objective function as penalty
        penalty = sum(weight * model.slack[s] for s in model.slack)
        obj = model.obj.expr + penalty
        sense = model.obj.sense
        model.del_component(model.obj)
        model.obj = pe.Objective(sense=sense, expr=obj)
        # Modify constraints to incorporate slacks
        for c in model.slack:
            # Deactivate hard constraints
            model.cons[c].deactivate()
            if model.cons[c].equality:
                # Equality: add +/- slack
                model.cons.add(model.cons[c].body + model.slack[c] >= model.cons[c].lower)
                model.cons.add(model.cons[c].body - model.slack[c] <= model.cons[c].upper)
            elif model.cons[c].lower is not None:
                # Lower bound: add + slack
                model.cons.add(model.cons[c].body + model.slack[c] >= model.cons[c].lower)
            else:
                # Upper bound: add - slack
                model.cons.add(model.cons[c].body - model.slack[c] <= model.cons[c].upper)
        return model_pen

    def first_solution_heuristic(self, nodes_limit=1):
        """Return a copy that stops after finding the first feasible solution."""
        # Clone model
        model_heur = self.clone()
        # Set solution limit
        if self.solver == "scip":
            model_heur.opt.options["limits/solutions"] = nodes_limit
        elif self.solver == "gurobi":
            model_heur.opt.options["SolutionLimit"] = nodes_limit
        else:
            raise ValueError("Solver '{}' does not support setting a solution limit.".format(self.solver))
        return model_heur

    def primal_heuristic(self, heuristic_name="rens"):
        """Return a copy configured to run a single primal heuristic."""
        # Clone model
        model_heur = self.clone()
        if self.solver == "scip":
            # Only process root node
            model_heur.opt.options["limits/nodes"] = 1
            # Disable presolve and separation
            model_heur.opt.options["presolving/maxrounds"] = 0
            model_heur.opt.options["separating/maxrounds"] = 0
            # All SCIP primal heuristics
            all_heuristics = [# Rounding
                              "rounding", "simplerounding", "randrounding", "zirounding",
                              # Shifting
                              "shifting", "intshifting",
                              # Flip
                              "oneopt", "twoopt",
                              # Indicator
                              "indicator",
                              # Diving
                              "actconsdiving", "indicatordiving", "farkasdiving",
                              "conflictdiving", "nlpdiving", "guideddiving",
                              "adaptivediving", "coefdiving", "pscostdiving",
                              "objpscostdiving", "fracdiving", "veclendiving",
                              "distributiondiving", "rootsoldiving",
                              "linesearchdiving", "intdiving",
                              # LNS
                              "alns", "localbranching", "rins", "rens", "gins",
                              "dins", "lpface", "clique", "crossover",
                              "mutation", "trustregion", "vbounds",
                              # Subsolve
                              "feaspump", "subnlp",
                              # Repair
                              "bound", "completesol", "fixandinfer", "repair",
                              # Other
                              "locks", "mpec", "multistart", "octane", "ofins",
                              "padm", "proximity", "trivial", "trivialnegation",
                              "trysol", "undercover", "zeroobj", "reoptsols"]
            if heuristic_name not in all_heuristics:
                raise ValueError(f"Unknown heuristic '{heuristic_name}'. Choose from {all_heuristics}.")
            # Disable all heuristics, then enable only the target one
            for heur in all_heuristics:
                model_heur.opt.options[f"heuristics/{heur}/freq"] = -1
            # freq=0 + freqofs=0: execute only at root node (depth 0)
            model_heur.opt.options[f"heuristics/{heuristic_name}/freq"] = 0
            model_heur.opt.options[f"heuristics/{heuristic_name}/freqofs"] = 0
            model_heur.opt.options[f"heuristics/{heuristic_name}/priority"] = 536870911
        else:
            raise ValueError("Solver '{}' does not support configuring primal heuristics.".format(self.solver))
        return model_heur
