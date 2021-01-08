# Repository for the Master Project - Physics informed neural networks for inverse problems in acoustic

## Description

Finite elements methods (FEMs) and Finite difference methods (FDMs) have always been used to solve partial differential equations (PDEs). In the recent years with the advance in computational power and the availability of software, artificial neural networks (ANN) can be used as a mesh free methods to approximate physical systems. Combining sparse or even no data, physics informed neural networks (PINNs) can be trained simultaneously on available data and the governing differential equations tofit a specific model, to compute the solution of an ordinary and partial differential equations or to identify a parameter (model inversion).

## Objectives

1. Implement the transient acoustic and elastic wave propagation in a FEM solver (Mathematica, Fenics or Esys) for homogeneous, layered and realistic earth model and save the outputs;
2. Use the available libraries (DeepXDE and SciANN) to implement FCNN surrogate models for solving the wave equation and code the same models from scratch using Pytorch;
3. Implement the surrogate models built with convolutional layers models presented in Geneva and Zabaras [2019], Zhu et al. [2019] and Zhu and Zabaras [2018] from scratch using PyTorch;
4. Compare the surrogate models outputs between FEM, libraries and PyTorch models and test their robustness by changing the domain (different soil layers) and the time interval (predictions outside their training range);
5. Test the inversion with the SciANN library and implement models from scratch with the same architecture as the ones presented in the papers;
6. Compare the inversion results between the FEM, the FCNNs and convolutional models;
7. Test the best performing models on real data instead of synthetic data generated with FEM.
