import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from model import *
from model_burger_pinn import *

import scipy.io # To load the matrix
import numpy as np
import os
import json
import copy

import warnings
warnings.filterwarnings("ignore")

# # # CUDA support 
if torch.cuda.is_available():
    device = torch.device('cuda')
else:
    device = torch.device('cpu')

# # # Known labels dataset
class u_Dataset(Dataset):
    def __init__(self, inputs, outputs):
        super(u_Dataset, self).__init__()

        self.inputs = inputs
        self.outputs = outputs

    def __len__(self):
        return self.inputs.shape[0]
        
    def __getitem__(self, index):

        # X 
        X = self.inputs[index, :]

        y = self.outputs[index]
        return X, y

# # # Residuals dataset

# # # Parameters
nu = 0.01/np.pi

N_u_vec = [0,50,100,200,500,1000,2000,5000,10000]
N_f = 0

# Load data 
data = scipy.io.loadmat('burgers_shock.mat')

for N_u in N_u_vec:

    # Epoch
    epoch = 50000
    
    # Model name
    dir_save = 'DATA/'
    model_name = f'B_E{epoch}_U{N_u}_F{N_f}'

    print("\n\tModel Name : {}\n".format(model_name))

    # # # Data
    t = data['t'].flatten()[:,None]
    x = data['x'].flatten()[:,None]
    Exact = np.real(data['usol']).T

    X, T = np.meshgrid(x,t)

    X_star = np.hstack((X.flatten()[:,None], T.flatten()[:,None]))
    u_star = Exact.flatten()[:,None]                

    # Initial conditions
    xx1 = np.hstack((X[0:1,:].T, T[0:1,:].T))  # x = -1:1, t = 0
    uu1 = Exact[0:1,:].T

    # Boundary conditions
    xx2 = np.hstack((X[:,0:1], T[:,0:1]))   # x = -1, t = 0:1
    uu2 = Exact[:,0:1]

    # Boundary conditions
    xx3 = np.hstack((X[:,-1:], T[:,-1:])) # x = 1, t = 0:1
    uu3 = Exact[:,-1:]

    X_u_train = np.vstack([xx1, xx2, xx3])
    u_train = np.vstack([uu1, uu2, uu3])

    # BC + IC : Always 100 points to enforce well possedness PDE
    idx = np.random.choice(X_u_train.shape[0], 100, replace=False)
    X_u_train = X_u_train[idx, :]
    u_train = u_train[idx,:]

    # Collocation points on the domain : range from 0 to 10'000
    idx = np.random.choice(X_star.shape[0], N_u, replace=False)
    X_c_train = X_star[idx,:]
    c_train = u_star[idx,:]

    # Add IC + BC with the collocation points sampled on the domain
    X_u_train = np.vstack([X_u_train, X_c_train])
    u_train = np.vstack([u_train, c_train])

    # Create datasets and dataloaders
    u_dataset = u_Dataset(X_u_train,u_train)

    batch_size = 100 if int(N_u/10) == 0 else int(N_u/10)

    u_trainloader = DataLoader(u_dataset, batch_size=batch_size, shuffle=True, num_workers=0)

    # # # Set up PINN model
    dnn = Model(in_size=2,neurons_layer=[20,20,20,20,20,20],out_size=1)

    # Check if file exists - if true then the training was aborted and we need to load the file with the existing model parameters and the remaining number of training iteration
    if os.path.isfile(dir_save + model_name): 
        checkpoint = torch.load(dir_save + model_name)
        dnn.load_state_dict(checkpoint['model_state_dict'])

        epoch = epoch - checkpoint['epoch']

        print('\n\tRemaining epoch :',epoch)
    else:
        print('\n\tRemaining epoch :',epoch)

    dnn = dnn.to(device)

    optimizer = torch.optim.Adam(dnn.parameters(), lr=1e-4)

    net_u = Net_U(model=dnn)

    pinn_loss = nn.MSELoss()

    # # # Training
    tot_loss = [] # list to store all the losses

    best_loss = 1e3 # Initial best loss

    for i in range(epoch):
        dnn.train()

        epoch_loss = 0

        for X_u, u in u_trainloader:
            
            X_u.requires_grad = True
            x_u = torch.tensor(X_u[:, 0:1]).float().to(device)
            t_u = torch.tensor(X_u[:, 1:2]).float().to(device)

            u = torch.tensor(u).float().to(device)     

            optimizer.zero_grad()

            preds_u = net_u(x_u,t_u)

            loss = pinn_loss(u,preds_u)

            loss.backward()

            optimizer.step()

            # Add loss
            epoch_loss += loss.item()

        tot_loss.append(epoch_loss)

        if epoch_loss < best_loss:
            best_loss = epoch_loss
            best_dnn = copy.deepcopy(dnn)

        if i % 100 == 0:
            print('Iter %d, Loss: %.3e' % (i,epoch_loss))
        
        if i % 1000 == 0:
            print('Saving model')
            PATH = dir_save + model_name + '.pt'
            torch.save({'epoch':i,'model_state_dict':best_dnn.state_dict()}, PATH)


    # # # Save model 
    PATH = dir_save + model_name + '.pt'
    torch.save(best_dnn.state_dict(), PATH)

    # # # Save list losses
    with open(PATH + '.txt', 'w') as f:
        json.dump(tot_loss, f)