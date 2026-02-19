"""
Loss function aggregators with sum-reduction over constraint violations.

Overrides neuromancer's default mean-reduction: for a violation tensor of
shape (batch, d1, d2, ...), each constraint's loss contribution becomes
``weight * mean_batch( sum_{d1,d2,...}(violation) )`` instead of
``weight * mean(violation)``.
"""

import math

import torch
import numpy as np
from neuromancer.loss import PenaltyLoss as _NMPenaltyLoss


class PenaltyLoss(_NMPenaltyLoss):
    """
    PenaltyLoss that sums constraint violations over non-batch
    dimensions instead of averaging.

    For a violation tensor of shape ``(batch, d1, d2, ...)``, the loss
    contribution of each constraint is::

        weight * mean_over_batch( sum_over_d1_d2_...(violation) )

    This is equivalent to the standard neuromancer ``PenaltyLoss`` with
    ``weight *= prod(d1, d2, ...)``, but avoids the need to manually
    scale penalty weights by constraint dimensionality.
    """

    def calculate_constraints(self, input_dict):
        loss = 0.0
        output_dict = {}
        C_values = []
        C_violations = []
        eq_flags = []
        for c in self.constraints:
            output = c(input_dict)
            output_dict = {**output_dict, **output}
            cvalue = output[c.output_keys[1]]
            cviolation = output[c.output_keys[2]]
            # sum over constraint dims, mean over batch
            flat = cviolation.reshape(cviolation.shape[0], -1)
            loss += c.weight * flat.sum(dim=1).mean()
            nr_constr = math.prod(cvalue.shape[1:])
            eq_flags += nr_constr * [str(c.comparator) == 'eq']
            C_values.append(cvalue.reshape(cvalue.shape[0], -1))
            C_violations.append(flat)
        if self.constraints:
            equalities_flags = np.array(eq_flags)
            C_violations = torch.cat(C_violations, dim=-1)
            C_values = torch.cat(C_values, dim=-1)
            output_dict['C_violations'] = C_violations
            output_dict['C_values'] = C_values
            output_dict['C_eq_violations'] = C_violations[:, equalities_flags]
            output_dict['C_ineq_violations'] = C_violations[:, ~equalities_flags]
            output_dict['C_eq_values'] = C_values[:, equalities_flags]
            output_dict['C_ineq_values'] = C_values[:, ~equalities_flags]
        output_dict['penalty_loss'] = loss
        return output_dict
