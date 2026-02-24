#!/usr/bin/env python
# coding: utf-8
"""
Submit experiments for MIRB
"""
import os
import argparse
import random
import numpy as np
import torch
from torch.utils.data import DataLoader
from reins import DictDataset
import experiments
import submitit

# Deterministic settings
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

# Config namespace
config = argparse.Namespace()

# Fixed hyperparameters
sizes = [1, 3, 10, 30, 100, 300, 1000, 3000, 10000]
hsize_dict = {1:4, 3:8, 10:16, 30:32, 100:64, 300:128, 1000:256, 3000:512, 10000:1024}
config.batch_size = 64                  # Batch size
config.hlayers_sol = 5                  # Number of hidden layers for solution mapping
config.hlayers_rnd = 4                  # Number of hidden layers for rounding network
config.lr = 1e-3                        # Learning rate
config.steepness = 50                   # Steepness factor
train_size = 8000                       # Number of train
test_size = 100                         # Number of test size
val_size = 1000                         # Number of validation size
penalty_weights = [2, 6, 20, 60, 200]


def is_done(csv_path):
    """Return True if result CSV exists and contains 100 data rows."""
    if not os.path.exists(csv_path):
        return False
    with open(csv_path) as f:
        return sum(1 for _ in f) > 100


def submit_job(func, *args, timeout_min, csv_path):
    """Submit a single SLURM job via submitit, skipping if result already exists."""
    if is_done(csv_path):
        print(f"        -> skip")
        return
    partition = "gpubase_bygpu_b1" if timeout_min <= 180 else "gpubase_bygpu_b2"
    executor = submitit.AutoExecutor(folder="logs")
    executor.update_parameters(
        slurm_additional_parameters={"account": "def-khalile2_gpu",
                                     "gres": "gpu:h100:1",
                                     "partition": partition,
                                     "exclude": "fc10512"},
        timeout_min=timeout_min,
        mem_gb=64,
        cpus_per_task=16,
    )
    job = executor.submit(func, *args)
    print(f"        Submitted job with ID: {job.job_id}")


# Parameters as input data
p_low, p_high = 1.0, 8.0
a_low, a_high = 0.5, 4.5

print("Rosenbrock\n")
for size in sizes:
    # Random seed per size for reproducibility
    random.seed(42)
    np.random.seed(42)
    torch.manual_seed(42)
    torch.cuda.manual_seed(42)

    # Set size-dependent config
    config.size = size
    config.hsize = hsize_dict[size]
    num_blocks = size

    # Generate data (shared across all penalties)
    p_train = torch.empty(train_size, 1).uniform_(p_low, p_high)
    p_test  = torch.empty(test_size, 1).uniform_(p_low, p_high)
    p_val   = torch.empty(val_size, 1).uniform_(p_low, p_high)
    a_train = torch.empty(train_size, num_blocks).uniform_(a_low, a_high)
    a_test  = torch.empty(test_size, num_blocks).uniform_(a_low, a_high)
    a_val   = torch.empty(val_size, num_blocks).uniform_(a_low, a_high)

    # Datasets
    data_train = DictDataset({"p":p_train, "a":a_train}, name="train")
    data_test = DictDataset({"p":p_test, "a":a_test}, name="test")
    data_val = DictDataset({"p":p_val, "a":a_val}, name="dev")

    # Torch dataloaders
    loader_train = DataLoader(data_train, config.batch_size, num_workers=0, collate_fn=data_train.collate_fn, shuffle=True, pin_memory=True)
    loader_test = DataLoader(data_test, config.batch_size, num_workers=0, collate_fn=data_test.collate_fn, shuffle=False, pin_memory=True)
    loader_val = DataLoader(data_val, config.batch_size, num_workers=0, collate_fn=data_val.collate_fn, shuffle=False, pin_memory=True)

    # Set timeout based on problem size
    timeout = 60 if size <= 300 else 360
    print(f"  Size: {size}")

    for penalty in penalty_weights:
        # Set penalty in config
        config.penalty = penalty
        print(f"    Penalty weight: {penalty}, timeout={timeout}min")

        # Projection versions
        config.project = True
        # Adaptive selection rounding + projection
        print("        Adaptive Selection, with projection")
        submit_job(experiments.rosenbrock.run_AS, loader_train, loader_test, loader_val, config,
                   timeout_min=timeout,
                   csv_path=f"result/stat/rb_cls{penalty}_{size}-p.csv")
        # Dynamic threshold rounding + projection
        print("        Dynamic Threshold, with projection")
        submit_job(experiments.rosenbrock.run_DT, loader_train, loader_test, loader_val, config,
                   timeout_min=timeout,
                   csv_path=f"result/stat/rb_thd{penalty}_{size}-p.csv")
        # STE rounding + projection
        print("        STE Rounding, with projection")
        submit_job(experiments.rosenbrock.run_RS, loader_train, loader_test, loader_val, config,
                   timeout_min=timeout,
                   csv_path=f"result/stat/rb_ste{penalty}_{size}-p.csv")

        print()
