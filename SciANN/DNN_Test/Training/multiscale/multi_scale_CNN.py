import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import cv2

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms

import os

from kornia.filters import SpatialGradient

if torch.cuda.is_available():
    device = 'cuda:0'
else:
    device = 'cpu'

    # # # Dataloader
class dataset(Dataset):
    def __init__(self, dir_path, csv_file):
        super(Dataset, self).__init__()

        self.dir_path = dir_path
        self.csv_file = pd.read_csv(csv_file,index_col=0)

        self.output = self.csv_file['y_number']

        self.transform = transforms.Compose([
    transforms.ToTensor()
])

    def __len__(self):
        return self.csv_file.shape[0]
        
    def __getitem__(self, index):

        # Get the indice for the output wavefield
        self.output_im = self.output.astype(int)[index]

        inputs = [cv2.imread(self.dir_path + f'Simple_Homogeneous_Moseley_Event0000_{im}.tiff',cv2.IMREAD_UNCHANGED) for im in range(self.output_im-4,self.output_im)]
        #print(list(im for im in range(self.output_im-4,self.output_im))) # For debugging

        outputs = [cv2.imread(self.dir_path + f'Simple_Homogeneous_Moseley_Event0000_{im}.tiff',cv2.IMREAD_UNCHANGED) for im in [self.output_im]]
        #print(list(im for im in [self.output_im])) # For debugging

        inputs = self.transform(np.array(inputs))
        outputs = self.transform(np.array(outputs))
        sample = {"wave_input": inputs,
                    "wave_input_label":self.output_im,
                    "wave_output": outputs,
                    "wave_output_label":self.output_im}
        return sample

class ConvBlock(nn.Module): 
    def __init__(self,in_channels): 
        super(ConvBlock, self).__init__()

        self.convblock = torch.nn.Sequential(
            torch.nn.Conv2d(in_channels=in_channels, out_channels=32, kernel_size=3, padding=1),
            torch.nn.ReLU(),
            torch.nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, padding=1),
            torch.nn.ReLU(),
            torch.nn.Conv2d(in_channels=64, out_channels=32, kernel_size=3, padding=1),
            torch.nn.ReLU(),
            torch.nn.Conv2d(in_channels=32, out_channels=1, kernel_size=3, padding=1),
        )

    def forward(self, x):
        x = self.convblock(x)
        return x

class MultiScale(nn.Module):
    def __init__(self,in_channels): 
        super(MultiScale, self).__init__()

        self.convblock_1 = ConvBlock(in_channels=in_channels)
        self.convblock_2 = ConvBlock(in_channels=in_channels+1)
        self.convblock_3 = ConvBlock(in_channels=in_channels+1)

        self.upsample = nn.Upsample(scale_factor=(2,2))

    def forward(self,x_4,x_2,x_1):

        x = self.convblock_1(x_4)
        x = self.upsample(x)

        x = self.convblock_2(torch.cat((x,x_2),dim=1))
        x = self.upsample(x)

        x = self.convblock_2(torch.cat((x,x_1),dim=1))

        return x

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

class MSLoss(nn.Module):
    def __init__(self): 
        super(MSLoss, self).__init__()

        self.gdl_fn = GDLLoss()
        self.mse_fn = nn.MSELoss()

    def forward(self,gen,gt):

        # Loss function  : L_2 (preds-true) + L_GDL (preds-true) + L_G (TODO)
        
        loss_l2 = self.mse_fn(gen,gt)
        loss_gdl = self.gdl_fn(gen,gt)

        loss = loss_l2 + loss_gdl

        return loss, loss_l2, loss_gdl

training_data = dataset('Simple_Homogeneous_Moseley/','Simple_Homogeneous_Moseley_Event0000_Continuous.csv')

# # # Training
dataloader = DataLoader(training_data, batch_size=2, shuffle=True)

model = MultiScale(in_channels=4)
model = model.to(device)

criterion = MSLoss()

optimizer = torch.optim.Adam(model.parameters(), lr=0.0001)

epochs = 1000

for epoch in range(epochs):
    
    train_loss = 0
    train_loss_l2 = 0
    train_loss_gdl = 0

    model.train()

    for samples in dataloader:

        wave_input = samples['wave_input'].transpose(2, 1)
        wave_input = wave_input.to(device)
        wave_input.require_grad = True

        #print(wave_input.size()) # For debugging

        x_4 = wave_input[:,:,::4,::4]
        x_2 = wave_input[:,:,::2,::2]
        x_1 = wave_input

        #print(x_4.size()) # For debugging
        #print(x_2.size()) # For debugging
        #print(x_1.size()) # For debugging
 
        gt = samples['wave_output'].transpose(2, 1)
        gt = gt.to(device)

        optimizer.zero_grad()

        gen = model(x_4,x_2,x_1)

        loss, loss_l2, loss_gdl = criterion(gen, gt)

        loss.backward()
        optimizer.step()

        train_loss += loss.item() * samples['wave_input'].size(0)
        train_loss_l2 += loss_l2.item() * samples['wave_input'].size(0)
        train_loss_gdl += loss_gdl.item() * samples['wave_input'].size(0)

    epoch_loss = train_loss / training_data.__len__()
    epoch_loss_l2 = train_loss_l2 / training_data.__len__()
    epoch_loss_gdl = train_loss_gdl / training_data.__len__()
    
    print('Epoch : %d, Loss : %.5e, Loss L2: %.5e, Loss GDL: %.5e' % (epoch,epoch_loss,epoch_loss_l2,epoch_loss_gdl))

# # # Save model
dir_save = 'L2_GDL/'
model_name = f'L2_GDL_E{epochs}'

if not os.path.exists(dir_save): 
    os.makedirs(dir_save)

PATH = dir_save + model_name + '.pt'
torch.save(model.state_dict(), PATH)