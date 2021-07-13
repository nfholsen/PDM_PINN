# PHYSICS INFORMED NEURAL NETWORKS FOR SURROGATE MODELLING AND INVERSE PROBLEMS IN GEOTECHNICS

## Description

Finite elements methods (FEMs) have benefited from decades of development to solve partial differential equations (PDEs) and to simulate physical systems. In the recent years, machine learning (ML) and artificial neural networks (ANN) have shown great potential to approximate such systems. Combining sparse or even no data, physics informed neural networks (PINNs) can be trained simultaneously on available data and the governing differential equations to fit a specific model or to compute the solution of an ordinary and partial differential equations.

This master project focuses on the implementation of ANN models in order to predict the evolution of an hyperbolic PDE that is the acoustic wave equation in geotechnic problems. Different model architectures and training scenarios are implemented to create a surrogate model that will predict from the first initial wavefields the acoustic propagation. The convolutional network with the U-Net architecture trained in a semi-supervised approach with a physical loss yields promising results, the lower average RMSE and the higher SSIM. This model is able to extrapolate the acoustic propagation on data not shown during its training but lacks of robustness when predicting wavefields at larger time than the ones provided for its training.

To cite this work if you use it or find it useful : 

```latex
@article{olsen2021pinn,
  title = {},
  journal = {},
  pages = {},
  year = {},
  issn = {},
  doi = {},
  url = {},
  author = {Nils Olsen and Brice Lecampion}
}
```

## Objectives and contributions

1. Compare different network architecture (MultiScale, Encoder and U-Net) with different training scenarios that are either data driven or physics constrained in order to see their abilities at predicting the acoustic wave equation;

![Alt Text](https://github.com/nfholsen/PDM_PINN/blob/master/Figures_README/WaveEq.svg)

2. Select the best model based on their performance on an homogeneous and layered domain to be the surrogate model. Test and assess the performance of the surrogate model on realistic experiment set up conducted at the Geo Energy Laboratory at EPFL; 
3. Create a huge dataset with multiple Salvus (https://doi.org/10.1093/gji/ggy469) experiments that can easily be extended for different fractures shapes, domain sizes and sources and receivers locations. 

## Files provided in the Repository 

1. Salvus files to generate the data for the experiments. Different media can be implemented : homogeneous, layered and cracked. For the cracked domain the user can specify the shape of the fracture : rectangular, circular and elliptical along with its size. Different shapes can be combined too. The inputs for the experiments need to be written in a .json file.
2. MLP to solve a set of artificial datasets in the `MLP_Test/Dummy` folder. This was implemented to test that a MLP could match any function in 2D and specifically unbalanced dataset.
2. The 1D Burgers equation is solved in the `MLP_Burgers_1D` folder and a comparison in made between the non physic and physics informed neural networks.
3. Convolutional Physics Informed Neural Networks with different training scenarios in the `DNN_Test` folder to simulate the wave equation. The models and the training methodologies are inspired by Geneva and Zabaras [[2019](https://arxiv.org/pdf/1906.05747.pdf)] for the Auto-Regressive training methodology and Weiqiang [[2017](https://doi.org/10.13140/RG.2.2.21994.77764)], Alguacil [[2020](https://doi.org/10.2514/6.2020-2513)], Fotiadis [[2020](https://www.sciencedirect.com/science/article/pii/S0022460X21003527)] and Sorteberg [[2018](https://arxiv.org/abs/1812.01609)] for the more classic supervised training. Clear metrics can be easily reproduced by training the models (around 2h on a NVidia Quattro P2000 GPU for 500 epochs), or by downloading the trained parameters and the results on the following link : 

### Results

**Homogeneous dataset** : Average on the training, space and time extrapolation

|      Scenarios     |       U-Net     |      Encoder    |     MultiScale    |
|:------------------:|:---------------:|:---------------:|:-----------------:|
|     AR PINN MSE    |     2.73e-01    |     2.60e-01    |      3.70e-01     |
|          L2        |     5.80e-01    |     7.88e-01    |      7.13e-01     |
|        L2 GDL      |     6.58e-01    |     8.08e-01    |      7.39e-01     |
|      L2 GDL MAE    |     4.95e-01    |     8.06e-01    |      8.26e-01     |
|       PINN MSE     |     4.83e-01    |     7.49e-01    |      7.11e-01     |
|       PINN RES     |     7.66e-01    |     7.34e-01    |      7.20e-01     |

Table 1 : SSIM

![Wavefields Homogeneous UNET PINN RES](https://github.com/nfholsen/PDM_PINN/blob/master/Figures_README/homogeneous_unet_PINN_RES.gif)

Figure 1 : Wavefields predictions for the homogeneous dataset

**Heterogeneous dataset** : Average on the training, space and time extrapolation

|      Scenarios     |       U-Net     |      Encoder    |     MultiScale    |
|:------------------:|:---------------:|:---------------:|:-----------------:|
|     AR PINN MSE    |     2.58e-01    |     5.55e-01    |      1.12e-01     |
|          L2        |     7.89e-01    |     6.73e-01    |      4.02e-01     |
|        L2 GDL      |     7.90e-01    |     6.81e-01    |      4.05e-01     |
|      L2 GDL MAE    |     7.82e-01    |     6.61e-01    |      2.62e-01     |
|       PINN MSE     |     7.54e-01    |     6.61e-01    |      3.67e-01     |
|       PINN RES     |     8.14e-01    |     6.28e-01    |      6.97e-01     |

Table 2 : SSIM

![Wavefields Heterogeneous UNET PINN RES](https://github.com/nfholsen/PDM_PINN/blob/master/Figures_README/heterogeneous_unet_PINN_RES.gif)

Figure 2 : Wavefields predictions for the heterogeneous dataset

**Surrogate model**

|                |              |       RMSE      |       SSIM      |     Relative Norm    |     Mean Abs Error    |     Median True / Max True    |
|:--------------:|:------------:|:---------------:|:---------------:|:--------------------:|:---------------------:|:-----------------------------:|
|     Crack 1    |     Train    |     9.41e+00    |     7.77e-01    |        6.32e-01      |        4.01e+00       |            4.39e-03           |
|     Crack 1    |      Test    |     1.64e+01    |     7.63e-01    |        1.11e+00      |        5.96e+00       |            3.44e-03           |
|     Crack 2    |     Train    |     1.51e+01    |     7.89e-01    |        1.01e+00      |        5.62e+00       |            3.82e-03           |
|     Crack 2    |      Test    |     1.71e+01    |     8.22e-01    |        1.14e+00      |        5.77e+00       |            2.99e-03           |

![Wavefields Surrogate Model 1 UNET PINN RES](https://github.com/nfholsen/PDM_PINN/blob/master/Figures_README/heterogeneous_crack_1_event0004_unet_PINN_RES.gif)

Figure 3 : Wavefields predictions on the small crack for the Event 4 (Train)

![Wavefields Surrogate Model 2 UNET PINN RES](https://github.com/nfholsen/PDM_PINN/blob/master/Figures_README/heterogeneous_crack_2_event0004_unet_PINN_RES.gif)

Figure 4 : Wavefields predictions on the big crack for the Event 4 (Test)


