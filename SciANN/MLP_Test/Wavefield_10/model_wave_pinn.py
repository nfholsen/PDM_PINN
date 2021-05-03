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

    def forward(self, x, y, t):
        """ The pytorch autograd version of calculating residual """
        u = self.net_u(x, y, t)
        
        u_t = torch.autograd.grad(
            u, t, 
            grad_outputs=torch.ones_like(u),
            retain_graph=True,
            create_graph=True
        )[0]
        u_tt = torch.autograd.grad(
            u_t, t, 
            grad_outputs=torch.ones_like(u_t),
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

        u_y = torch.autograd.grad(
            u, y, 
            grad_outputs=torch.ones_like(u),
            retain_graph=True,
            create_graph=True
        )[0]
        u_yy = torch.autograd.grad(
            u_y, y, 
            grad_outputs=torch.ones_like(u_y),
            retain_graph=True,
            create_graph=True
        )[0]
        
        f = u_xx + u_yy - 1/self.nu * u_tt
        return f

class Net_U(nn.Module):
    def __init__(self,model):

        nn.Module.__init__(self)

        self.model = model

    def forward(self, x, y, t):
        u = self.model(torch.cat([x, y, t], dim=1))
        return u

class PINN_loss(nn.Module):
    def __init__(self):

        nn.Module.__init__(self)

    def forward(self,u,u_pred,f_pred):
        loss_u = torch.mean((u - u_pred) ** 2)
        loss_f = torch.mean(f_pred ** 2)

        loss = loss_u + loss_f
        return loss, loss_u, loss_f