# self-pruning-neural-network-cifar10
This case study presents the design and implementation of a self-pruning neural network for image classification on the CIFAR-10 dataset using PyTorch. Traditional neural networks are often over-parameterized, leading to increased computational cost and inefficiency. 

# Overview
This project implements a self-pruning neural network that dynamically removes unimportant connections during training using learnable gate parameters and L1 sparsity regularization. The model is evaluated on the CIFAR-10 dataset to study the trade-off between accuracy and model compression.

# Technologies Used
Python
PyTorch
NumPy
Matplotlib

# Model Architecture
Input (3072)
   ↓
PrunableLinear (1024) + BatchNorm + ReLU
   ↓
PrunableLinear (512) + BatchNorm + ReLU
   ↓
PrunableLinear (256) + BatchNorm + ReLU
   ↓
Output Layer (10 classes)

# Implementation
Custom PrunableLinear layer with learnable gate parameters
Sigmoid-based gating mechanism for weight pruning
L1 sparsity regularization to enforce pruning
Training using Adam optimizer and cosine learning rate scheduler

# Results
🔹 Gate Distribution

Shows how weights are pruned based on gate values.


![Gate Distribution](results/gate_distribution.png)
🔹 Training Curves

Displays accuracy and sparsity progression during training.

![Training Curves](results/training_curves.png)

# Observations
Increasing sparsity strength (λ) increases pruning
Higher pruning may slightly reduce accuracy
Balanced λ provides optimal performance

# How to Run
1. Install Dependencies
pip install torch torchvision matplotlib numpy
2. Run Code
python src/main.py

# Project Structure
self-pruning-neural-network/
│
├── README.md
│
├── src/
│   └── main.py
│
├── results/
│   ├── gate_distribution.png
│   └── training_curves.png
│
├── docs/
│   └──Report.pdf
