#!/bin/bash
ENV_NAME="image_classifier"

PYTHON_VERSION="3.9"

echo "================================="
echo "Image Sharpness Classification Project Environment Setup"
echo "================================="

if ! command -v conda &> /dev/null; then
    echo "ERROR: conda not found!"
    echo "Please install Anaconda or Miniconda"
    exit 1
fi

echo "Conda is installed"

if conda env list | grep -q "^${ENV_NAME}"; then
    echo "WARNING: Environment '${ENV_NAME}' already exists"
    read -p "Do you want to delete the existing environment and recreate it? (y/N): " -n 1 -r
    echo
    if [[ $REPLY =~ ^[Yy]$ ]]; then
        echo " Deleting the existing environment..."
        conda env remove -n ${ENV_NAME} -y
    else
        echo "Exiting without changes."
        exit 1
    fi
fi

echo "Creating new conda environment: ${ENV_NAME} (Python ${PYTHON_VERSION})"
conda create -n ${ENV_NAME} python=${PYTHON_VERSION} -y

if [ $? -ne 0 ]; then
    echo "Environment creation failed"
    exit 1
fi

echo "Creating new conda environment: ${ENV_NAME} (Python ${PYTHON_VERSION})"

echo "Activating environment..."
source $(conda info --base)/etc/profile.d/conda.sh
conda activate ${ENV_NAME}

if [ $? -ne 0 ]; then
    echo "Environment activation failed"
    exit 1
fi

echo "Environment activated successfully"

echo " Installing core dependencies (via conda)..."

conda install -c conda-forge -y \
    numpy \
    scipy \
    tqdm \
    matplotlib \
    seaborn \
    pandas \
    jupyter \
    notebook \
    ipykernel

if [ $? -ne 0 ]; then
    echo "failed to install core dependencies"
    exit 1
fi

echo "conda dependencies installed successfully"

echo "installing OpenCV (via pip)..."
pip install opencv-python

if [ $? -ne 0 ]; then
    echo "failed to install OpenCV"
    exit 1
fi

echo "OpenCV installed successfully"

read -p "Do you want to install development dependencies (pytest, black, flake8)? (y/N): " -n 1 -r
echo
if [[ $REPLY =~ ^[Yy]$ ]]; then
    echo "Installing development dependencies..."
    conda install -c conda-forge -y pytest pytest-cov black flake8
    echo "Development dependencies installed successfully"
fi

echo "Verifying installation..."
python -c "
import cv2
import numpy as np
import scipy
import tqdm
import matplotlib
import seaborn
import pandas
print('Environment verification successful!')
print(f'Python version: {cv2.__version__}')
print(f'OpenCV version: {cv2.__version__}')
print(f'NumPy version: {np.__version__}')
print(f'SciPy version: {scipy.__version__}')
print(f'Matplotlib version: {matplotlib.__version__}')
"

if [ $? -eq 0 ]; then
    echo "Environment installation completed!"
    echo ""
    echo "Usage:"
    echo "  conda activate ${ENV_NAME}"
    echo "  python main.py --help"
    echo ""
    echo "Environment information:"
    echo "  Environment name: ${ENV_NAME}"
    echo "  Python version: ${PYTHON_VERSION}"
    echo "  Installation path: $(conda info --envs | grep ${ENV_NAME} | awk '{print $2}')"
    echo ""
    echo "Note: OpenCV is installed via pip to avoid conda/pip package management conflicts"
else
    echo "Environment verification failed"
    exit 1
fi