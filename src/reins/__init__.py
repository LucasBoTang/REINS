"""
reins: Neuromancer extension for Mixed-Integer Nonlinear Programming.
"""

# ---- Re-exports from neuromancer ----
from neuromancer.dataset import DictDataset
from neuromancer.constraint import Objective, Constraint
from neuromancer.loss import PenaltyLoss
from neuromancer.problem import Problem
from neuromancer.trainer import Trainer

# ---- reins modules ----
from reins.blocks import MLPBnDrop
from reins.variable import Variable, VarType, TypeVariable
from reins.solver import LearnableSolver

__all__ = [
    # neuromancer re-exports
    "DictDataset", "Trainer",
    "Variable", "Objective", "Constraint", "PenaltyLoss", "Problem",
    # reins modules
    "MLPBnDrop",
    "VarType", "TypeVariable",
    "LearnableSolver",
]
