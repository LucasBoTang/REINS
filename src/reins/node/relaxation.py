"""
Relaxation node with output splitting by variable sizes.
"""

import torch
from neuromancer.system import Node


class RelaxationNode(Node):
    """
    Relaxation node that auto-splits network output by variable sizes.

    Args:
        callable: Network whose output dim equals sum of variable sizes.
        params: Parameter Variable or list of parameter Variables.
        vars: Decision Variable or list of decision Variables.
        sizes: Split sizes for multi-variable output.
        name: Module name.
    """

    def __init__(self, callable, params, vars, sizes=None, name="relaxation"):
        # Normalize to lists
        if not isinstance(params, (list, tuple)):
            params = [params]
        if not isinstance(vars, (list, tuple)):
            vars = [vars]

        # Derive keys from Variable objects
        param_keys = [p.key for p in params]
        output_keys = [v.key + "_rel" for v in vars]

        # Validate sizes for multi-variable output
        if len(output_keys) > 1 and sizes is None:
            raise ValueError("sizes is required for multi-variable output.")

        # Initialize base class
        super().__init__(callable, param_keys, output_keys, name=name)
        self.sizes = sizes

        # Validate sizes against network output dimension at init time
        if sizes is not None:
            out_dim = getattr(callable, 'out_features', None) or getattr(callable, 'outsize', None)
            if out_dim is not None and sum(sizes) != out_dim:
                raise ValueError(
                    f"Sum of sizes {sum(sizes)} != network output dim {out_dim}."
                )

    def forward(self, data):
        # Concatenate inputs if multiple keys
        inputs = [data[k] for k in self.input_keys]
        x = torch.cat(inputs, dim=-1) if len(inputs) > 1 else inputs[0]
        # Call the network
        output = self.callable(x)
        # Initialize result dict
        result = {}
        # Single output: no split needed
        if len(self.output_keys) == 1:
            result[self.output_keys[0]] = output
        # Multiple variables: split by sizes
        else:
            # Check that output dimension
            if sum(self.sizes) != output.shape[-1]:
                raise ValueError("Sum of sizes must equal output dimension.")
            # Split output by sizes and assign to corresponding keys
            offset = 0
            for size, key in zip(self.sizes, self.output_keys):
                result[key] = output[:, offset:offset + size]
                offset += size
        return result
