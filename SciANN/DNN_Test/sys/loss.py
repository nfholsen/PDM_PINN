import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.modules.utils import _quadruple

from kornia.filters import SpatialGradient
import numpy as np

####################
# # # GDL Loss # # #
####################

class GDLLoss(nn.Module):
    def __init__(self,alpha=2): 
        super(GDLLoss, self).__init__()

        self.grad = SpatialGradient(normalized=True)
        self.alpha = alpha

    def forward(self,gen,gt,alpha=2):

        self.gen_dx = self.grad(gen)[:,:,0,:,:]
        self.gen_dy = self.grad(gen)[:,:,1,:,:]
        self.gt_dx = self.grad(gt)[:,:,0,:,:]
        self.gt_dy = self.grad(gt)[:,:,1,:,:]

        self.grad_diff_x = torch.abs(self.gt_dx - self.gen_dx)
        self.grad_diff_y = torch.abs(self.gt_dy - self.gen_dy)

        return torch.mean(self.grad_diff_x ** self.alpha + self.grad_diff_y ** self.alpha)

class PINNLoss_RES(nn.Module):
    def __init__(self, dh=5, dt=0.002, c=2200, device='cuda:0'):
       super(PINNLoss_RES, self).__init__()

       # Parameters of the mesh
       self.dh = dh
       self.dt = dt
       self.c = c
       self.device = device

       self.P = (self.dt * self.c/self.dh) ** 2

       # Kernels filters
       # x FD
       self.weight_h = torch.FloatTensor([[[[ 0,  0,  0],[ 1, -2,  1],[ 0,  0,  0]]]]).to(self.device)
       # y FD
       self.weight_v = torch.FloatTensor([[[[ 0,  1,  0],[ 0, -2,  0],[ 0,  1,  0]]]]).to(self.device) 
       self.padding = _quadruple(1)

    def forward(self,fem_data):     

       u_n_preds = fem_data[:,-1:,:,:] # Predicitons by the NN
       u_n = fem_data[:,-2:-1,:,:]
       u_n_1 = fem_data[:,-3:-2,:,:]

       # Compute derivatives
       self.u_dxdx = F.conv2d(F.pad(u_n, self.padding, mode='circular'), self.weight_h, stride=1, padding=0, bias=None)
       self.u_dydy = F.conv2d(F.pad(u_n, self.padding, mode='circular'), self.weight_v, stride=1, padding=0, bias=None)   

       # PDE acoustic wave equation : 
       residuals = self.P * self.u_dxdx + self.P * self.u_dydy + 2 * u_n - u_n_1 - u_n_preds

       return torch.norm(residuals)


class PINNLoss_MSE(nn.Module):
    def __init__(self, dh=5, dt=0.002, c=2200, device='cuda:0'):
        super(PINNLoss_MSE, self).__init__()

        # Parameters of the mesh
        self.dh = dh
        self.dt = dt
        self.c = c
        self.device = device

        self.P = (self.dt ** 2) * (self.c ** 2)/(self.dh ** 2)

        # Kernels filters
        # x FD
        self.weight_h = torch.FloatTensor([[[[ 0,  0,  0],[ 1, -2,  1],[ 0,  0,  0]]]]).to(self.device)

        # y FD
        self.weight_v = torch.FloatTensor([[[[ 0,  1,  0],[ 0, -2,  0],[ 0,  1,  0]]]]).to(self.device) 

        self.padding = _quadruple(1)


    def forward(self,inputs):
        """
        inputs : combination of inputs and outputs of the model, shape = (Minibatch,Channels,Width,Height)
        """    

        # TODO : assert length of channels is 5

        preds = inputs[:,-1:,:,:] # last from inputs (prediction from the NN)
        u_n = inputs[:,-2:-1,:,:] 
        u_n_1 = inputs[:,-3:-2,:,:]

        # Compute derivatives
        self.u_dxdx = F.conv2d(F.pad(u_n, self.padding, mode='circular'), self.weight_h, stride=1, padding=0, bias=None)
        self.u_dydy = F.conv2d(F.pad(u_n, self.padding, mode='circular'), self.weight_v, stride=1, padding=0, bias=None)

        # Next time step computed based on the finite difference method for the wave equation : 
        next_u = self.P * self.u_dxdx + self.P * self.u_dydy + 2 * u_n - u_n_1

        return torch.mean( torch.square(preds - next_u) )

class MSLoss(nn.Module):
    def __init__(self, *loss_list): 
        super(MSLoss, self).__init__()

        self.loss_list = nn.ModuleList([*loss_list])

    def forward(self,gen=None,gt=None,pinn=None):
        
        loss_output = []
        for loss in self.loss_list:
            if loss.forward.__code__.co_argcount == 2: # If the number of inputs is 1, not Preds & True
                loss_output.append(loss(pinn))  # For physics loss
            else : 
                loss_output.append(loss(gen,gt)) # For supervised loss

        # TODO : change tuple if only one loss with if condition
        loss_output = [sum(loss_output)] + loss_output # Combine elements : first sum of losses for back prop, then all the individual losses
        return tuple(loss_output)