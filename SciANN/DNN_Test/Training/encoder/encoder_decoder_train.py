import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import cv2

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

# # # Dataloader
class dataset(Dataset):
    def __init__(self, dir_path, csv_file,transform=None):
        super(Dataset, self).__init__()

        self.dir_path = dir_path
        self.csv_file = pd.read_csv(csv_file,index_col=0)

        self.input = self.csv_file['x']
        self.output = self.csv_file['y']


        self.transform = transform

    def __len__(self):
        return self.csv_file.shape[0]
        
    def __getitem__(self, index):

        # X 
        x = self.input.iloc[index]
        y = self.output.iloc[index]

        wave_input = cv2.imread(self.dir_path + x, cv2.IMREAD_UNCHANGED)[None, :, :]
        wave_output = cv2.imread(self.dir_path + y, cv2.IMREAD_UNCHANGED)[None, :, :]
        if self.transform:
            wave_input = self.transform(wave_input)
            wave_output = self.transform(wave_output)

        sample = {"wave_input": wave_input,
                    "wave_input_label":index,
                    "wave_output": wave_output,
                    "wave_output_label":index+1,}
        return sample

training_data = dataset('Simple_Homogeneous_Moseley/','Simple_Homogeneous_Moseley_Event0000.csv')

dataloader = DataLoader(training_data, batch_size=16, shuffle=True)

class Autoencoder(torch.nn.Module): 
    def __init__(self): 
        super(Autoencoder, self).__init__()
        self.encoder = torch.nn.Sequential(
            torch.nn.Conv2d(in_channels=1, out_channels=32, kernel_size=4, stride=2),
            torch.nn.ReLU(),
            torch.nn.Conv2d(in_channels=32, out_channels=64, kernel_size=4, stride=2),
            torch.nn.ReLU(),
            torch.nn.Conv2d(in_channels=64, out_channels=128, kernel_size=4, stride=2),
        )

        self.decoder = torch.nn.Sequential( 
            torch.nn.ConvTranspose2d(in_channels=128, out_channels=64, kernel_size=5, stride=2),
            torch.nn.ConvTranspose2d(in_channels=64, out_channels=32, kernel_size=5, stride=2),
            torch.nn.ConvTranspose2d(in_channels=32, out_channels=1, kernel_size=4, stride=2)
        )

    def forward(self, x):
        x = self.encoder(x)

        x = self.decoder(x)
        return x

if torch.cuda.is_available():
    device = 'cuda:0'
else:
    device = 'cpu'

model = Autoencoder()

criterion = nn.MSELoss()

optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

epochs = 1000

model = model.to(device)

for epoch in range(epochs):
    
    train_loss = 0

    model.train()

    for samples in dataloader:
        x_train = samples['wave_input'].to(device)
        y_train = samples['wave_output'].to(device)

        optimizer.zero_grad()

        y_preds = model(x_train)

        loss = criterion(y_preds, y_train)

        loss.backward()
        optimizer.step()

        train_loss += loss.item() * samples['wave_input'].size(0)

    epoch_loss = train_loss / training_data.__len__()
    print('Epoch : {}, Loss : {}'.format(epoch,epoch_loss))

# Save model

PATH = 'Encoder_Decoder_1000.pt'
torch.save(model.state_dict(), PATH)