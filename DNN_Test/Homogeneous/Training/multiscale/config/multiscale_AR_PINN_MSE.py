import sys
import json

sys_path = json.load(open('../../../../../paths.json',))['sys_path']

sys.path.append(sys_path)

from loss import *
from multi_scale import MultiScale
from dataloader import *
from BaseModel import BaseModel

import torch.optim as optim
import logging

if torch.cuda.is_available():
    device = 'cuda:0'
else:
    device = 'cpu'

##################
# # # Config # # # 
##################

epochs = 500
batch_size = 1

# Data
data_dir = '../../../../Training_Data/Moseley_Homogeneous/'
data_csv = '../../../../Training_Data/Moseley_Homogeneous_AR.csv'

# Paths
save_dir = '../results/'
save_pt_best = f'Best_AR_PINN_MSE_E{epochs}.pt'
save_pt = f'AR_PINN_MSE_E{epochs}.pt'
save_txt = f'AR_PINN_MSE_E{epochs}.yml'

checkpoint_path= f'checkpoint_AR_PINN_MSE_E{epochs}.pt'

# # # Data
training_data = dataset(data_dir,data_csv)
train_loader = DataLoader(training_data, batch_size=batch_size, shuffle=True)

net = MultiScale(in_channels=4)

# Optimizer & Scheduler
optimizer = optim.Adam(net.parameters(), lr=0.00001, weight_decay=1e-6)

class MultiScaleModel(BaseModel):
    def __init__(self, net, N,T, opt=None, sched=None, logger=None, print_progress=True, device='cuda:0'):
        """

        """
        super().__init__(net, opt, sched, logger, print_progress, device)

        self.loss_fn = PINNLoss_MSE(dh=5, dt=0.002, c=2500, device='cuda:0')

        self.T = T
        self.N = N

    def forward_loss(self, data):
        """

        """
        input = data['wave_input'].transpose(2, 1)
        input = input.to(self.device)

        loss_epoch = 0

        for i in range(self.N):
            loss = 0
            self.optimizer.zero_grad()

            for j in range(0,self.T):
        
                input_4 = input[:,:,::4,::4]
                input_2 = input[:,:,::2,::2]
                input_1 = input

                output = self.net(input_4,input_2,input_1) # 1 x 4 x 300 x 300  --> 1 x 1 x 300 x 300

                pinn = torch.cat((input,output),axis=1) # 1 x 5 x 300 x 300 

                loss = loss + self.loss_fn(inputs=pinn)

                input = pinn[:,1:,:,:] # 1 x 4 x 300 x 300 

            #print('N =',i)

            loss_epoch = loss_epoch + loss

            loss.backward()
            self.optimizer.step()
            input = input.detach()

        return torch.tensor(0., requires_grad=True), {'Loss':loss_epoch, 'Loss AR PINN MSE':loss_epoch} 

# Create the model
model = MultiScaleModel(net=net, N=50,T=4, opt=optimizer, sched=None, logger=None, print_progress=True, device=device)

# Train the model
model.train(epochs, train_loader, checkpoint_path=checkpoint_path, checkpoint_freq=5, save_best=None)

# Save
#model.save_best(export_path=save_dir + save_pt_best)
model.save(export_path=save_dir + save_pt)
model.save_outputs(export_path=save_dir + save_txt)