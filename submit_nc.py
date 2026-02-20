#!/usr/bin/env python
# coding: utf-8
"""
Submit experiments for INC
"""
import argparse
import random
import numpy as np
import torch
from torch.utils.data import DataLoader
from sklearn.model_selection import train_test_split
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
sizes = [5, 10, 20, 50, 100, 200, 500, 1000]
hsize_dict = {5:16, 10:32, 20:64, 50:128, 100:256, 200:512, 500:1024, 1000:2048}
config.batch_size = 64                  # Batch size
config.hlayers_sol = 5                  # Number of hidden layers for solution mapping
config.hlayers_rnd = 4                  # Number of hidden layers for rounding network
config.lr = 1e-3                        # Learning rate
train_size = 8000                       # Number of train
test_size = 1000                        # Number of test size
val_size = 1000                         # Number of validation size
num_data = train_size + test_size + val_size


def submit_job(func, *args, timeout_min):
    """Submit a single SLURM job via submitit."""
    executor = submitit.AutoExecutor(folder="logs")
    executor.update_parameters(
        slurm_additional_parameters={"account": "def-khalile2",
                                     "constraint": "v100l"},
        timeout_min=timeout_min,
        mem_gb=64,
        cpus_per_task=16,
        gpus_per_node=1,
    )
    job = executor.submit(func, *args)
    print(f"Submitted job with ID: {job.job_id}")


print("Simple Non-Convex")
for size in sizes:
    # Random seed per size for reproducibility
    random.seed(42)
    np.random.seed(42)
    torch.manual_seed(42)
    torch.cuda.manual_seed(42)

    # Set size-dependent config
    config.size = size
    config.hsize = hsize_dict[size]
    num_var = size
    num_ineq = size

    # Data sample from uniform distribution
    b_samples = torch.from_numpy(np.random.uniform(-1, 1, size=(num_data, num_ineq))).float()
    d_samples = torch.from_numpy(np.random.uniform(-0.1, 0.1, size=(num_data, num_ineq))).float()

    # Data split
    ind = list(range(num_data))
    ind_train, ind_test = train_test_split(ind, test_size=test_size, random_state=42, shuffle=True)
    ind_train, ind_val = train_test_split(ind_train, test_size=val_size, random_state=42, shuffle=True)
    # Convert indices to tensors for efficient indexing
    itrain, itest, ival = torch.tensor(ind_train), torch.tensor(ind_test), torch.tensor(ind_val)
    data_train = DictDataset({"b": b_samples[itrain], "d": d_samples[itrain]}, name="train")
    data_test = DictDataset({"b": b_samples[itest], "d": d_samples[itest]}, name="test")
    data_val = DictDataset({"b": b_samples[ival], "d": d_samples[ival]}, name="dev")
    # Torch dataloaders
    loader_train = DataLoader(data_train, config.batch_size, num_workers=4,
                              collate_fn=data_train.collate_fn, shuffle=True,
                              pin_memory=True, persistent_workers=True)
    loader_test  = DataLoader(data_test, config.batch_size, num_workers=4,
                              collate_fn=data_test.collate_fn, shuffle=False,
                              pin_memory=True, persistent_workers=True)
    loader_val   = DataLoader(data_val, config.batch_size, num_workers=4,
                              collate_fn=data_val.collate_fn, shuffle=False,
                              pin_memory=True, persistent_workers=True)

    # Set timeout based on problem size
    timeout = 20 if size <= 50 else 60
    print(f"Submitting size={size}, timeout={timeout}min")

    # Non-projection versions
    config.project = False
    # Adaptive selection rounding
    submit_job(experiments.nonconvex.run_AS, loader_train, loader_test, loader_val, config,
               timeout_min=timeout)
    # Dynamic threshold rounding
    submit_job(experiments.nonconvex.run_DT, loader_train, loader_test, loader_val, config,
               timeout_min=timeout)
    # Learn-then-round
    submit_job(experiments.nonconvex.run_LR, loader_train, loader_test, loader_val, config,
               timeout_min=timeout)
    # STE rounding
    submit_job(experiments.nonconvex.run_RS, loader_train, loader_test, loader_val, config,
               timeout_min=timeout)

    # Projection versions
    config.project = True
    # Adaptive selection rounding + projection
    submit_job(experiments.nonconvex.run_AS, loader_train, loader_test, loader_val, config,
               timeout_min=timeout)
    # Dynamic threshold rounding + projection
    submit_job(experiments.nonconvex.run_DT, loader_train, loader_test, loader_val, config,
               timeout_min=timeout)
    # STE rounding + projection
    submit_job(experiments.nonconvex.run_RS, loader_train, loader_test, loader_val, config,
               timeout_min=timeout)
