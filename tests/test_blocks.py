"""
Unit tests for MLPBnDrop network block.
"""

import pytest
import torch
import torch.nn as nn

from reins.blocks import MLPBnDrop


class TestMLPBnDrop:
    """Test MLPBnDrop with various configurations."""

    def test_forward_shape(self):
        net = MLPBnDrop(insize=20, outsize=10, hsizes=[64, 64])
        x = torch.randn(8, 20)
        y = net(x)
        assert y.shape == (8, 10)

    def test_default_params(self):
        net = MLPBnDrop(insize=10, outsize=5, hsizes=[32, 32])
        layer_types = [type(m) for m in net.net]
        assert nn.BatchNorm1d in layer_types
        assert nn.Dropout in layer_types
        assert nn.ReLU in layer_types

    def test_no_bnorm(self):
        net = MLPBnDrop(insize=10, outsize=5, hsizes=[32], bnorm=False)
        layer_types = [type(m) for m in net.net]
        assert nn.BatchNorm1d not in layer_types

    def test_no_dropout(self):
        net = MLPBnDrop(insize=10, outsize=5, hsizes=[32], dropout=0.0)
        layer_types = [type(m) for m in net.net]
        assert nn.Dropout not in layer_types

    def test_no_bnorm_no_dropout(self):
        net = MLPBnDrop(insize=10, outsize=5, hsizes=[32],
                        bnorm=False, dropout=0.0)
        layer_types = [type(m) for m in net.net]
        assert nn.BatchNorm1d not in layer_types
        assert nn.Dropout not in layer_types
        # Should only have Linear and ReLU layers
        for m in net.net:
            assert isinstance(m, (nn.Linear, nn.ReLU))

    def test_custom_nonlin(self):
        net = MLPBnDrop(insize=10, outsize=5, hsizes=[32], nonlin=nn.Tanh)
        layer_types = [type(m) for m in net.net]
        assert nn.Tanh in layer_types
        assert nn.ReLU not in layer_types

    def test_bias_false(self):
        net = MLPBnDrop(insize=10, outsize=5, hsizes=[32], bias=False)
        for m in net.net:
            if isinstance(m, nn.Linear):
                assert m.bias is None

    def test_single_hidden(self):
        net = MLPBnDrop(insize=10, outsize=5, hsizes=[32])
        linears = [m for m in net.net if isinstance(m, nn.Linear)]
        assert len(linears) == 2  # hidden + output
        assert linears[0].in_features == 10
        assert linears[0].out_features == 32
        assert linears[1].in_features == 32
        assert linears[1].out_features == 5

    def test_multiple_hidden(self):
        net = MLPBnDrop(insize=10, outsize=5, hsizes=[64, 32, 16])
        linears = [m for m in net.net if isinstance(m, nn.Linear)]
        assert len(linears) == 4  # 3 hidden + 1 output
        assert linears[0].in_features == 10
        assert linears[0].out_features == 64
        assert linears[1].in_features == 64
        assert linears[1].out_features == 32
        assert linears[2].in_features == 32
        assert linears[2].out_features == 16
        assert linears[3].in_features == 16
        assert linears[3].out_features == 5

    def test_train_eval_mode(self):
        net = MLPBnDrop(insize=10, outsize=5, hsizes=[32])
        x = torch.randn(8, 10)
        # Train mode: dropout active, outputs vary
        net.train()
        out_train_1 = net(x)
        out_train_2 = net(x)
        # Eval mode: dropout disabled, outputs deterministic
        net.eval()
        out_eval_1 = net(x)
        out_eval_2 = net(x)
        assert torch.equal(out_eval_1, out_eval_2)

    def test_forward_multi_input(self):
        """forward(*inputs) should concatenate multiple tensors."""
        net = MLPBnDrop(insize=7, outsize=3, hsizes=[16])
        net.eval()
        a = torch.randn(4, 3)
        b = torch.randn(4, 4)
        # Multi-input call
        out_multi = net(a, b)
        # Single-input call with manual cat
        out_single = net(torch.cat([a, b], dim=-1))
        assert torch.equal(out_multi, out_single)

    def test_empty_hsizes(self):
        """hsizes=[] should produce direct linear mapping (no hidden layers)."""
        net = MLPBnDrop(insize=4, outsize=3, hsizes=[])
        linears = [m for m in net.net if isinstance(m, nn.Linear)]
        assert len(linears) == 1  # only output layer
        assert linears[0].in_features == 4
        assert linears[0].out_features == 3
        # Forward should work
        x = torch.randn(2, 4)
        y = net(x)
        assert y.shape == (2, 3)

    def test_out_features_attribute(self):
        """out_features attribute should equal outsize."""
        net = MLPBnDrop(insize=10, outsize=7, hsizes=[32])
        assert net.out_features == 7

    def test_export(self):
        import reins
        assert hasattr(reins, "MLPBnDrop")
        assert "MLPBnDrop" in reins.__all__
        assert reins.MLPBnDrop is MLPBnDrop


class TestMLPBnDropNumerical:
    """Verify exact numerical behavior of MLPBnDrop."""

    def test_zero_weight_outputs_bias(self):
        """Network with zero weights should output bias regardless of input."""
        net = MLPBnDrop(insize=3, outsize=2, hsizes=[], bias=True)
        # Zero weights, set known bias
        with torch.no_grad():
            linear = [m for m in net.net if isinstance(m, nn.Linear)][0]
            linear.weight.zero_()
            linear.bias.copy_(torch.tensor([1.5, -2.0]))
        net.eval()
        x = torch.randn(4, 3)
        y = net(x)
        expected = torch.tensor([1.5, -2.0]).expand(4, -1)
        assert torch.allclose(y, expected)

    def test_identity_like_no_hidden(self):
        """hsizes=[] with identity weight computes y = Wx + b."""
        net = MLPBnDrop(insize=2, outsize=2, hsizes=[], bias=True)
        with torch.no_grad():
            linear = [m for m in net.net if isinstance(m, nn.Linear)][0]
            linear.weight.copy_(torch.eye(2))
            linear.bias.zero_()
        net.eval()
        x = torch.tensor([[3.0, -1.0], [0.0, 5.0]])
        y = net(x)
        assert torch.allclose(y, x)

    def test_multi_input_numerical(self):
        """Multi-input forward with known weights."""
        net = MLPBnDrop(insize=3, outsize=1, hsizes=[], bias=True)
        with torch.no_grad():
            linear = [m for m in net.net if isinstance(m, nn.Linear)][0]
            linear.weight.copy_(torch.tensor([[1.0, 2.0, 3.0]]))
            linear.bias.copy_(torch.tensor([0.0]))
        net.eval()
        a = torch.tensor([[1.0]])
        b = torch.tensor([[2.0, 3.0]])
        y = net(a, b)
        # 1*1 + 2*2 + 3*3 = 14
        assert torch.allclose(y, torch.tensor([[14.0]]))
