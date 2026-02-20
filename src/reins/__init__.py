"""
reins: Neuromancer extension for Mixed-Integer Nonlinear Programming.
"""

__version__ = "0.0.1"

# ---- Re-exports from neuromancer ----
from neuromancer.dataset import DictDataset
from neuromancer.problem import Problem
from neuromancer.trainer import Trainer

# ---- REINS modules ----
from reins.blocks import MLPBnDrop
from reins.variable import Variable, VarType, TypeVariable
from reins.loss import PenaltyLoss
from reins.solver import LearnableSolver

__all__ = [
    # Neuromancer re-exports
    "DictDataset", "Trainer", "Problem",
    # REINS modules
    "MLPBnDrop",
    "Variable", "VarType", "TypeVariable",
    "PenaltyLoss", "LearnableSolver"
]
