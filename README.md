# Repository for the Master Project - PHYSICS INFORMED NEURAL NETWORKS FOR SURROGATE MODELLING AND INVERSE PROBLEMS IN GEOTECHNICS

## Description

Finite elements methods (FEMs) have benefited from decades of development to solve partial differential equations (PDEs) and to simulate physical systems. In the recent years, machine learning (ML) and artificial neural networks (ANN) have shown great potential to approximate such systems. Combining sparse or even no data, physics informed neural networks (PINNs) can be trained simultaneously on available data and the governing differential equations to fit a specific model or to compute the solution of an ordinary and partial differential equations.

This master project focuses on the implementation of ANN models in order to predict the evolution of an hyperbolic PDE that is the acoustic wave equation in geotechnic problems. Different model architectures and training scenarios are implemented to create a surrogate model that will predict from the first initial wavefields the acoustic propagation. The U-Net model trained in a semi-supervised approach with a physical loss yields promising results, the lower average RMSE and the higher SSIM. This model is able to extrapolate the acoustic propagation on data not shown during its training but lacks of robustness when predicting wavefields at larger time than the ones provided for its training.


## Wave Equation 

$\rho \nabla \cdot \left( \frac{1}{\rho} \nabla p \right)- \frac{1}{\nu^2} \frac{\partial^2p}{\partial t^2} = - \rho \frac{\partial^2 f}{\partial t^2}$

## Objectives

1. Implement the transient acoustic and elastic wave propagation in a FEM solver (Mathematica, Fenics or Esys) for homogeneous, layered and realistic earth model and save the outputs;
2. Use the available libraries (DeepXDE and SciANN) to implement FCNN surrogate models for solving the wave equation and code the same models from scratch using Pytorch;
3. Implement the surrogate models built with convolutional layers models presented in Geneva and Zabaras [2019], Zhu et al. [2019] and Zhu and Zabaras [2018] from scratch using PyTorch;
4. Compare the surrogate models outputs between FEM, libraries and PyTorch models and test their robustness by changing the domain (different soil layers) and the time interval (predictions outside their training range);
5. Test the inversion with the SciANN library and implement models from scratch with the same architecture as the ones presented in the papers;
6. Compare the inversion results between the FEM, the FCNNs and convolutional models;
7. Test the best performing models on real data instead of synthetic data generated with FEM.
