#!/usr/bin/env python
# coding: utf-8
"""
Shared utilities for experiment pipelines.
"""

import time
import os
import numpy as np
import pandas as pd
import torch


def set_seeds(seed=42):
    """Set random seeds for reproducibility."""
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def make_result_df(params, sols, viols, objvals, mean_viols, max_viols, num_viols, elapseds, proj_iters=None):
    """Create a standardized result DataFrame."""
    return pd.DataFrame({
        "Param": params, "Sol": sols, "Viol": viols, "Obj Val": objvals,
        "Mean Violation": mean_viols, "Max Violation": max_viols,
        "Num Violations": num_viols,
        "Num Proj Iters": proj_iters if proj_iters is not None else [0] * len(objvals),
        "Elapsed Time": elapseds,
    })


def print_summary(df, show_unsolved=False, sleep=False):
    """Print experiment summary statistics."""
    if sleep:
        time.sleep(1)
    print(df.describe())
    print("Number of infeasible solutions: {}".format(np.sum(df["Num Violations"] > 0)))
    if show_unsolved:
        print("Number of unsolved instances: ", df["Sol"].isna().sum())


def save_results(df, npz_path, csv_path):
    """Save solution arrays to .npz and summary statistics to .csv."""
    os.makedirs("result/sol", exist_ok=True)
    os.makedirs("result/stat", exist_ok=True)
    np.savez_compressed(
        npz_path,
        Param=np.array(df["Param"].tolist()),
        Sol=np.array(df["Sol"].tolist(), dtype=object),
        Viol=np.array(df["Viol"].tolist(), dtype=object),
    )
    df[["Obj Val", "Mean Violation", "Max Violation", "Num Violations", "Num Proj Iters", "Elapsed Time"]].to_csv(csv_path)


def record_viol(model, viols, mean_viols, max_viols, num_viols):
    """Record constraint violation statistics from a solved model."""
    viol = model.cal_violation()
    viols.append(viol.tolist())
    mean_viols.append(np.mean(viol))
    max_viols.append(np.max(viol))
    num_viols.append(np.sum(viol > 1e-6))


def record_failure(sols, viols, objvals, mean_viols, max_viols, num_viols):
    """Record None entries for a failed solve."""
    sols.append(None)
    viols.append(None)
    objvals.append(None)
    mean_viols.append(None)
    max_viols.append(None)
    num_viols.append(None)
