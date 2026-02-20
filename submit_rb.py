#!/usr/bin/env python
# coding: utf-8
"""
Submit experiments for MIRB
"""
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

# Set parser
parser = argparse.ArgumentParser()
parser.add_argument("--penalty",
                    type=float,
                    default=100,
                    help="Penalty weight")
config = parser.parse_args()

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


def submit_job(func, *args, timeout_min):
    """Submit a single SLURM job via submitit."""
    partition = "gpubase_bygpu_b1" if timeout_min <= 180 else "gpubase_bygpu_b2"
    executor = submitit.AutoExecutor(folder="logs")
    executor.update_parameters(
        slurm_additional_parameters={"account": "def-khalile2_gpu",
                                     "gres": "gpu:h100:1",
                                     "partition": partition},
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

    # Generate data
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
    print(f"    Submitting size={size}, timeout={timeout}min")

    # Non-projection versions
    config.project = False
    # Adaptive selection rounding
    print("        Adaptive Selection, no projection")
    submit_job(experiments.rosenbrock.run_AS, loader_train, loader_test, loader_val, config,
               timeout_min=timeout)
    # Dynamic threshold rounding
    print("        Dynamic Threshold, no projection")
    submit_job(experiments.rosenbrock.run_DT, loader_train, loader_test, loader_val, config,
               timeout_min=timeout)
    # Learn-then-round
    print("        Learn-then-Round, no projection")
    submit_job(experiments.rosenbrock.run_LR, loader_train, loader_test, loader_val, config,
               timeout_min=timeout)
    # STE rounding
    print("        STE Rounding, no projection")
    submit_job(experiments.rosenbrock.run_RS, loader_train, loader_test, loader_val, config,
               timeout_min=timeout)

    # Projection versions
    config.project = True
    # Adaptive selection rounding + projection
    print("        Adaptive Selection, with projection")
    submit_job(experiments.rosenbrock.run_AS, loader_train, loader_test, loader_val, config,
               timeout_min=timeout)
    # Dynamic threshold rounding + projection
    print("        Dynamic Threshold, with projection")
    submit_job(experiments.rosenbrock.run_DT, loader_train, loader_test, loader_val, config,
               timeout_min=timeout)
    # STE rounding + projection
    print("        STE Rounding, with projection")
    submit_job(experiments.rosenbrock.run_RS, loader_train, loader_test, loader_val, config,
               timeout_min=timeout)

    print()