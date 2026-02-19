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
        input_keys: Parameter keys to read from data dict.
        output_keys: Variable key (str) or list of variable keys.
            Output keys are auto-appended with "_rel".
        sizes: Split sizes for multi-variable output.
        name: Module name.
    """

    def __init__(self, callable, input_keys, output_keys, sizes=None, name="relaxation"):
        # Normalize output keys
        if isinstance(output_keys, str):
            output_keys = [output_keys]

        # Output keys are auto-appended with "_rel"
        output_keys = [k + "_rel" for k in output_keys]

        # Validate sizes for multi-variable output
        if len(output_keys) > 1 and sizes is None:
            raise ValueError("sizes is required for multi-variable output.")

        # Initialize base class
        super().__init__(callable, input_keys, output_keys, name=name)
        self.sizes = sizes

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
