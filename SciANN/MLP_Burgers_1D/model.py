import torch
import torch.nn as nn

class Model(nn.Module):
    def __init__(self,
                in_size = 3, 
                neurons_layer=[100, 100],
                out_size = 1
                ):
        """
        Initializing the PINN
        """
        nn.Module.__init__(self)

        self.layers = nn.ModuleList()

        # first layer
        self.layers.append(nn.Linear(in_features=in_size, out_features=neurons_layer[0])) 

        # hidden layers
        for n_in, n_out in zip(neurons_layer[:-1], neurons_layer[1:]):
            self.layers.append(nn.Linear(in_features=n_in, out_features=n_out))

        # output layer
        self.out  = nn.Linear(in_features=neurons_layer[-1], out_features=out_size)
        
        self.tanh = nn.Tanh()

    def forward(self, x):
        """
        Forward Pass
        """
        for linear in self.layers:
            x = self.tanh(linear(x))
        x = self.out(x)
        return x