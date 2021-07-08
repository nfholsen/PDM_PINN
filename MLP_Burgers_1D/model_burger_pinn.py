import torch
import torch.nn as nn
from model import *

class Net_F(nn.Module):
    def __init__(self, net_u, nu):
        """
        Compute the residuals

        Inputs :
            net_u : 
            nu : 
            device : self explenatory, cuda if possible 
        """
        nn.Module.__init__(self)
        self.net_u = net_u
        self.nu = nu

    def forward(self, x, t):
        """ The pytorch autograd version of calculating residual """
        u = self.net_u(x, t)
        
        u_t = torch.autograd.grad(
            u, t, 
            grad_outputs=torch.ones_like(u),
            retain_graph=True,
            create_graph=True
        )[0]
        u_x = torch.autograd.grad(
            u, x, 
            grad_outputs=torch.ones_like(u),
            retain_graph=True,
            create_graph=True
        )[0]
        u_xx = torch.autograd.grad(
            u_x, x, 
            grad_outputs=torch.ones_like(u_x),
            retain_graph=True,
            create_graph=True
        )[0]
        
        f = u_t + u * u_x - self.nu * u_xx
        return f

class Net_U(nn.Module):
    def __init__(self,model):

        nn.Module.__init__(self)

        self.model = model

    def forward(self,x,t):
        u = self.model(torch.cat([x, t], dim=1))
        return u

class PINN_loss(nn.Module):
    def __init__(self):

        nn.Module.__init__(self)

    def forward(self,u,u_pred,f_pred):
        loss_u = torch.mean((u - u_pred) ** 2)
        loss_f = torch.mean(f_pred ** 2)

        loss = loss_u + loss_f
        return loss, loss_u, loss_f