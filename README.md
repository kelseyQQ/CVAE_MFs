# CVAE_MFs

Conditional Variational Autoencoder (CVAE) for porous microstructure generation and inverse design using Minkowski Functionals (MFs) as conditioning variables.

## Overview

This repository contains a CNN-based Conditional Variational Autoencoder for generating binary porous microstructures conditioned on three Minkowski Functionals:

- **M0**: porosity / volume fraction
- **M1**: interfacial measure / perimeter-related descriptor
- **M2**: Euler characteristic / topology descriptor

The project focuses on learning the relationship between morphology descriptors and microstructure realizations, and then using the trained model for **inverse design** and **design-space exploration**.

## Main functionalities

- Train a CNN-based CVAE on binary microstructure data
- Condition generation on target Minkowski Functionals
- Optimize hyperparameters using **Optuna**
- Evaluate generation quality using reconstructed and generated samples
- Explore the feasible inverse-design domain using an **octree-based boundary search**
- Export 2D/3D visualizations of the discovered design space

## Data format

The dataset loader expects .dat files with the following structure:

- Lines 0–39: 40 × 40 binary image (True / False)
- Lines 40–42: local Minkowski Functionals
- Line 43: global Minkowski Functionals (mean of local MFs)
- Line 44: sigma values (standard deviation of local MFs)

Each sample is loaded as:

- image tensor: shape [1, 40, 40]
- local MF values
- global MF values
- sigma values

## Dataset
The dataset used in this project is derived from the RockMicro Minkowski database developed by Sijmen Zwarts:
https://github.com/sfzwarts/RockMicro_Minkowski/tree/main/Data

## Model

The main model is implemented in CNN_CVAE.py.
It includes:

- residual downsampling blocks for the encoder
- residual upsampling blocks for the decoder
- a conditional VAE variant (CVAE_encoder)
- conditioning on 3-dimensional MF vectors

The training setup in this repository uses:

- grayscale input images
- condition_dim = 3
- convolutional base channels ch = 16
- block structure (1, 2, 4)
- batch normalization
- BCE-based reconstruction learning with KL regularization

## Dependencies
This project uses the following libraries:
- torch
- numpy
- matplotlib
- scikit-image
- scipy
- scikit-learn
- optuna 
- pandas
- seaborn
- plotly
- pyvista

pip install torch numpy matplotlib scikit-image scipy scikit-learn optuna pandas seaborn plotly pyvista

## How to use
1. Prepare the dataset
Unzip the dataset archives so that the expected folders are available.
2. Train the model with Optuna: Run python BO.py (Hyperparameter tuning is performed using [Optuna](https://github.com/optuna/optuna) [Optuna](https://optuna.org).)
3. Visualize Optuna results: Run python optuna_plot.py
4. Run inverse-design boundary exploration. After training, place the selected checkpoint at: exploratory_3D/model_best.pth.Then run: python inverse_design_octree.py

## Expected outputs

Depending on the script, outputs may include:

- trained model checkpoints
- reconstruction history plots
- generated image samples
- metrics CSV files
- violin plots for MF comparison
- Optuna study database / summary
- inverse-design boundary figures
- interactive HTML visualizations of the MF design space

## Visualisation of output

![heatmap](CVAE_MFs/optuna_plot/tpe_ratio_heatmap.png)
