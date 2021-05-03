import torch
import torch.nn as nn
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

class PINNLoss(nn.Module):
    def __init__(self, dh=5, dt=0.002, c=2200,device='cuda:0'):
        super(PINNLoss, self).__init__()

        # Parameters of the mesh
        self.dh = dh
        self.dt = dt
        self.c = c
        self.device = device

        # Kernels filters 
        # x
        self.x = np.array([[[ 0,  0,  0],
               [ 0,  0,  0],
               [ 0,  0,  0]],
              [[ 0,  0,  0],
               [ 1, -2,  1],
               [ 0,  0,  0]],
              [[ 0,  0,  0],
               [ 0,  0,  0],
               [ 0,  0,  0]]
               ])
        self.x = self.x/(self.dh ** 2)

        # y
        self.y = np.array([[[ 0,  0,  0],
               [ 0,  0,  0],
               [ 0,  0,  0]],
              [[ 0,  1,  0],
               [ 0, -2,  0],
               [ 0,  1,  0]],
              [[ 0,  0,  0],
               [ 0,  0,  0],
               [ 0,  0,  0]]
               ])
        self.y = self.y/(self.dh ** 2)

        # z
        self.z = np.array([[[ 0,  0,  0],
               [ 0,  1,  0],
               [ 0,  0,  0]],
              [[ 0,  0,  0],
               [ 0, -2,  0],
               [ 0,  0,  0]],
              [[ 0,  0,  0],
               [ 0,  1,  0],
               [ 0,  0,  0]]
               ])
        self.z = self.z/(self.dt ** 2)

        # u_dxdx
        self.conv_x = nn.Conv3d(3, 1, kernel_size=3, stride=1, padding=(0,1,1), bias=False)
        self.conv_x.weight=nn.Parameter(torch.from_numpy(self.x).float().unsqueeze(0).unsqueeze(0))
        self.conv_x = self.conv_x.to(self.device)
        # u_dydy
        self.conv_y = nn.Conv3d(3, 1, kernel_size=3, stride=1, padding=(0,1,1), bias=False)
        self.conv_y.weight=nn.Parameter(torch.from_numpy(self.y).float().unsqueeze(0).unsqueeze(0))
        self.conv_y = self.conv_y.to(self.device)

        # u_dtdt
        self.conv_z = nn.Conv3d(3, 1, kernel_size=3, stride=1, padding=(0,1,1), bias=False)
        self.conv_z.weight=nn.Parameter(torch.from_numpy(self.z).float().unsqueeze(0).unsqueeze(0))
        self.conv_z = self.conv_z.to(self.device)

    def forward(self,inputs):        

        # Compute derivatives
        u_dxdx, u_dydy, u_dtdt = self.conv_x(inputs), self.conv_y(inputs), self.conv_z(inputs)

        # PDE acoustic wave equation : 
        residuals = u_dxdx + u_dydy - u_dtdt * (1 / (self.c ** 2))

        return torch.norm(residuals)

class MSLoss(nn.Module):
    def __init__(self, *loss_list): 
        super(MSLoss, self).__init__()

        self.loss_list = nn.ModuleList([*loss_list])

    def forward(self,gen,gt,inputs=None):
        
        loss_output = []
        for loss in self.loss_list:
            if loss.forward.__code__.co_argcount == 2: # If the number of inputs is 1, not Preds & True
                loss_output.append(loss(inputs))
            else : 
                loss_output.append(loss(gen,gt))

        # TODO : change tuple if only one loss with if condition
        loss_output = [sum(loss_output)] + loss_output # Combine elements : first sum of losses for back prop, then all the individual losses
        return tuple(loss_output)