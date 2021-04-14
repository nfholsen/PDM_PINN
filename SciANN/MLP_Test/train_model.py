# Import libraries

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

import h5py

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

from model import *

# Pytorch Model Name
model_name = 'Moseley_PINN_1000.pt'

# Load Salvus data
path = "../NoCrack/Simple_Homogeneous_Moseley/Event0000/output.h5"
hf = h5py.File(path, 'r')

# Train set
max_timestep = 40

df_train = pd.DataFrame()

for ts in range(0,max_timestep,4):
    df_ts = pd.DataFrame()
    df_ts['X'] = hf['coordinates_ACOUSTIC'][:,0][:,0].astype(float)
    df_ts['Y'] = hf['coordinates_ACOUSTIC'][:,0][:,1].astype(float)
    df_ts['T'] = ts
    df_ts['True'] = hf['volume']['phi'][ts][:,0].mean(axis=1)

    df_train = pd.concat((df_train,df_ts),axis=0)
df_train = df_train.reset_index(drop=True)

# Test set
timestep = 36

df_test = pd.DataFrame()
df_test['X'] = hf['coordinates_ACOUSTIC'][:,0][:,0].astype(float)
df_test['Y'] = hf['coordinates_ACOUSTIC'][:,0][:,1].astype(float)
df_test['T'] = timestep
df_test['True'] = hf['volume']['phi'][timestep][:,0].mean(axis=1)

# # # CUDA support 
if torch.cuda.is_available():
    device = torch.device('cuda')
else:
    device = torch.device('cpu')

print(device)

# # # Model

class dataset(Dataset):
    def __init__(self, inputs, outputs):
        super(Dataset, self).__init__()

        self.inputs = inputs
        self.outputs = outputs

    def __len__(self):
        return self.inputs.shape[0]
        
    def __getitem__(self, index):

        # X 
        X = self.inputs[index, :]

        y = self.outputs[index]
        return X, y

def train(model,X_train,y_train,loss_fn,optimizer,epoch):
    
    model = model.to(device)

    train_dataset = dataset(X_train,y_train)

    trainloader = DataLoader(train_dataset, batch_size=10000, shuffle=True, num_workers=0)

    tot_loss = []

    for i in range(epoch):
        model.train()

        epoch_loss = 0

        for X, true in trainloader:

            x = torch.tensor(X[:, 0:1], requires_grad=True).float().to(device)
            y = torch.tensor(X[:, 1:2], requires_grad=True).float().to(device)
            t = torch.tensor(X[:, 2:3], requires_grad=True).float().to(device)

            true = torch.tensor(true).float().to(device)

            optimizer.zero_grad()

            preds = model(torch.cat([x, y, t], dim=1))

            loss = loss_fn(preds,true)

            loss.backward()
            optimizer.step()

            epoch_loss += loss

        tot_loss.append(epoch_loss)

        if i % 10 == 0:
                print('Iter %d, Loss: %.5e' % (i,epoch_loss))

    return model, tot_loss

def prediction(model,X_test,y_test,device):

    model = model.to(device)
    x = torch.tensor(X_test[:, 0:1], requires_grad=True).float().to(device)
    y = torch.tensor(X_test[:, 1:2], requires_grad=True).float().to(device)
    t = torch.tensor(X_test[:, 2:3], requires_grad=True).float().to(device)

    true = torch.tensor(y_test).float().to(device)

    model.eval()
    with torch.no_grad():

        preds = model(torch.cat([x, y, t], dim=1))

    preds = preds.detach().cpu().numpy()
    return preds

def plots(df_data):
    # Compute error :
    df_data['Error'] = df_data['Preds']-df_data['True']

    # Plot :
    fig, ax = plt.subplots(1,3,figsize=(16,4))

    im = ax[0].imshow(df_data.pivot_table(values='Preds',index='Y',columns='X').sort_index(axis=0,ascending=False))
    ax[0].set_title('Prediction')
    fig.colorbar(im, ax=ax[0])

    im = ax[1].imshow(df_data.pivot_table(values='True',index='Y',columns='X').sort_index(axis=0,ascending=False))
    ax[1].set_title('True')
    fig.colorbar(im, ax=ax[1])

    im = ax[2].imshow(df_data.pivot_table(values='Error',index='Y',columns='X').sort_index(axis=0,ascending=False))
    ax[2].set_title('Error')
    fig.colorbar(im, ax=ax[2])

    plt.show()

# Train 

model = Model(in_size=3)

loss_fn = nn.MSELoss()

optimizer = torch.optim.Adam(model.parameters(),lr = 0.001)

X_train = df_train.loc[:,['X','Y','T']].values
y_train = df_train.loc[:,['True']].values

epoch = 1000
model_train, loss_train = train(model,X_train,y_train,loss_fn,optimizer,epoch)

# Save model
PATH = model_name

torch.save(model_train.state_dict(), PATH)

# Load model
PATH = model_name

model_test = Model()
model_test.load_state_dict(torch.load(PATH))
model_test.eval()

preds = prediction(model_test,X_test,y_test,device)
df_test['Preds'] = preds

plots(df_test)