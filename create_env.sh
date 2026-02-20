#!/bin/bash
# coding: utf-8

# global vars
VENV_NAME=".venv"
PYTHON_VER="3.11"
CUDA_VER="12.2"
NM_VER="1.5.2"

# load module
echo "Load module..."
module purge
module load StdEnv/2023
module load gcc/12.3
module load python/$PYTHON_VER
module load cuda/$CUDA_VER
module load flexiblas

# create virtual env
if [ ! -d "./$VENV_NAME" ]; then
  echo "Create venv..."
  python -m venv $VENV_NAME
  source $VENV_NAME/bin/activate
  echo ""

  echo "Install requirements..."
  pip install --upgrade pip

  # PyTorch (CUDA version from CC wheelhouse)
  pip install --no-index torch torchvision torchaudio

  # neuromancer dependencies (installed before neuromancer to avoid scs/cvxpy issue)
  pip install numpy scipy matplotlib pandas scikit-learn
  pip install networkx tqdm dill
  pip install plum-dispatch==1.7.3
  pip install pydot==1.4.2
  pip install pyts
  pip install lightning wandb

  # neuromancer without cvxpy/scs dependency
  pip install neuromancer==$NM_VER --no-deps

  # job submission
  pip install submitit

  # reins package
  pip install -e . --no-deps

# activate virtual env
else
  echo "Activate venv..."
  source $VENV_NAME/bin/activate
fi
echo ""
