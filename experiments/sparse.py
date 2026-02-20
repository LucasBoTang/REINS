#!/usr/bin/env python
# coding: utf-8
"""
Sparse constraint matrix mask generation.

Generates a binary mask with two structural components:
  - Global rows (1/10): sparse rows spanning all variables (5% density)
  - Block-diagonal rows (9/10): 10 blocks with Dirichlet-sampled sizes (20% density)
"""

import numpy as np


def sparse_mask(num_rows, num_cols, rng):
    """
    Generate a sparse binary mask for a constraint matrix.

    Args:
        num_rows: Number of constraint rows.
        num_cols: Number of variable columns.
        rng: np.random.RandomState instance (will be advanced in-place).

    Returns:
        Binary mask array of shape (num_rows, num_cols).
    """
    mask = np.zeros((num_rows, num_cols), dtype=np.float32)

    # Split rows: 1/10 global, 9/10 block-diagonal
    num_global = max(1, num_rows // 10)
    num_block_rows = num_rows - num_global

    # Global sparse rows with 5% density
    mask[:num_global] = (rng.random((num_global, num_cols)) < 0.05).astype(np.float32)

    # Block-diagonal rows
    num_blocks = 10

    # Two independent Dirichlet(alpha=0.5) for row and column partitions
    row_props = rng.dirichlet(np.ones(num_blocks) * 1.0)
    col_props = rng.dirichlet(np.ones(num_blocks) * 1.0)

    # Convert proportions to integer sizes, ensuring at least 1 row/col per block
    row_sizes = _proportions_to_sizes(row_props, num_block_rows)
    col_sizes = _proportions_to_sizes(col_props, num_cols)

    # Fill block-diagonal with 20% density
    row_offsets = np.zeros(num_blocks + 1, dtype=int)
    col_offsets = np.zeros(num_blocks + 1, dtype=int)
    row_offsets[0] = num_global
    col_offsets[0] = 0
    for i in range(num_blocks):
        row_offsets[i + 1] = row_offsets[i] + row_sizes[i]
        col_offsets[i + 1] = col_offsets[i] + col_sizes[i]

    for i in range(num_blocks):
        rs, cs = row_sizes[i], col_sizes[i]
        block = (rng.random((rs, cs)) < 0.2).astype(np.float32)
        # Ensure at least one non-zero entry per block
        if block.sum() == 0:
            block[rng.randint(rs), rng.randint(cs)] = 1.0
        mask[row_offsets[i]:row_offsets[i + 1],
             col_offsets[i]:col_offsets[i + 1]] = block

    # Copy 1% of each block's columns to the next block (inter-block coupling)
    for i in range(num_blocks - 1):
        cs = col_sizes[i]
        num_copy = max(1, int(np.ceil(0.01 * cs)))
        copy_cols = rng.choice(cs, size=num_copy, replace=False) + col_offsets[i]
        next_rs = row_sizes[i + 1]
        for c in copy_cols:
            mask[row_offsets[i + 1]:row_offsets[i + 2], c] = (
                rng.random(next_rs) < 0.2
            ).astype(np.float32)

    return mask


def _proportions_to_sizes(proportions, total):
    """
    Convert Dirichlet proportions to integer sizes summing exactly to *total*.

    Each output size is at least 1.
    """
    # Ensure total is large enough to allocate at least 1 per block
    n = len(proportions)
    if total < n:
        raise ValueError(
            f"total ({total}) must be >= num_blocks ({n}) "
            f"to guarantee at least 1 per block."
        )
    # Initial floor rounding + 1 to ensure minimum of 1
    remaining = total - n
    raw = proportions * remaining
    sizes = np.floor(raw).astype(int) + 1  # +1 guarantees minimum of 1

    # Distribute leftover from floor rounding
    deficit = total - sizes.sum()
    if deficit > 0:
        fractional = raw - np.floor(raw)
        indices = np.argsort(-fractional)
        for j in range(deficit):
            sizes[indices[j]] += 1

    # Final sanity check
    assert sizes.sum() == total, f"sizes sum {sizes.sum()} != total {total}"
    assert (sizes >= 1).all(), f"some block has size 0: {sizes}"
    return sizes
