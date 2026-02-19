"""
Unit tests for src package structure and neuromancer re-exports.
"""

import pytest


class TestPackageImport:
    """Test that the package and subpackages can be imported."""

    def test_import_src(self):
        import reins

    def test_import_rounding_subpackage(self):
        import reins.rounding

    def test_import_projection_subpackage(self):
        import reins.projection

    def test_import_utils_subpackage(self):
        import reins.utils


class TestNeuromancerReExports:
    """Test that neuromancer components are re-exported correctly."""

    def test_node(self):
        from reins import Node
        from neuromancer.system import Node as Node_orig
        assert Node is Node_orig

    def test_dict_dataset(self):
        from reins import DictDataset
        from neuromancer.dataset import DictDataset as DD_orig
        assert DictDataset is DD_orig

    def test_trainer(self):
        from reins import Trainer
        from neuromancer.trainer import Trainer as Trainer_orig
        assert Trainer is Trainer_orig

    def test_objective(self):
        from reins import Objective
        from neuromancer.constraint import Objective as Obj_orig
        assert Objective is Obj_orig

    def test_constraint(self):
        from reins import Constraint
        from neuromancer.constraint import Constraint as Con_orig
        assert Constraint is Con_orig

    def test_penalty_loss(self):
        from reins import PenaltyLoss
        from neuromancer.loss import PenaltyLoss as PL_orig
        assert PenaltyLoss is PL_orig

    def test_problem(self):
        from reins import Problem
        from neuromancer.problem import Problem as Prob_orig
        assert Problem is Prob_orig

    def test_all_exports_listed(self):
        import reins
        expected = [
            "Node", "DictDataset", "Trainer",
            "Objective", "Constraint", "PenaltyLoss", "Problem",
            "MLPBnDrop", "VarType", "variable",
            "GradientProjection", "LearnableSolver",
        ]
        for name in expected:
            assert name in reins.__all__, f"{name} missing from __all__"
            assert hasattr(reins, name), f"{name} not accessible"
