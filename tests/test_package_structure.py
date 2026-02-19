"""
Unit tests for src package structure and neuromancer re-exports.
"""

import pytest


class TestPackageImport:
    """Test that the package and subpackages can be imported."""

    def test_import_src(self):
        import reins

    def test_import_node_subpackage(self):
        import reins.node

    def test_import_rounding_subpackage(self):
        import reins.node.rounding

    def test_import_projection_subpackage(self):
        import reins.projection


class TestNeuromancerReExports:
    """Test that neuromancer components are re-exported correctly."""

    def test_dict_dataset(self):
        from reins import DictDataset
        from neuromancer.dataset import DictDataset as DD_orig
        assert DictDataset is DD_orig

    def test_trainer(self):
        from reins import Trainer
        from neuromancer.trainer import Trainer as Trainer_orig
        assert Trainer is Trainer_orig

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
            "DictDataset", "Trainer",
            "PenaltyLoss", "Problem",
            "MLPBnDrop", "VarType", "TypeVariable", "Variable",
            "LearnableSolver",
        ]
        for name in expected:
            assert name in reins.__all__, f"{name} missing from __all__"
            assert hasattr(reins, name), f"{name} not accessible"
