import sys
import json

sys_path = json.load(open('../../../../../paths.json',))['sys_path']

sys.path.append(sys_path)

from loss import *
from encoder import AutoEncoder
from dataloader import *
from BaseModel import BaseModel

import torch.optim as optim

if torch.cuda.is_available():
    device = 'cuda:0'
else:
    device = 'cpu'

##################
# # # Config # # # 
##################

epochs = 500
batch_size = 4

# Data
data_dir = '../../../../Training_Data/Moseley_EARTH/'
data_csv = '../../../../Training_Data/Moseley_Earth.csv'
velocity_field = '../../../../Training_Data/Velocity_Field_1.npy'

# Paths
save_dir =  '../results/'
save_pt = f'PINN_MSE_E{epochs}.pt'
save_txt = f'PINN_MSE_E{epochs}.yml'

checkpoint_path = f'checkpoint_PINN_MSE_E{epochs}.pt'

# # # Data
training_data = dataset(data_dir,data_csv,velocity_field=velocity_field)
train_loader = DataLoader(training_data, batch_size=batch_size, shuffle=True)

net = AutoEncoder(in_channels=5,out_channels=1)

# Optimizer & Scheduler
optimizer = optim.Adam(net.parameters(), lr=0.0001, weight_decay=1e-6)

class AutoEncoderModel(BaseModel):
    def __init__(self, net, opt=None, sched=None, logger=None, print_progress=True, device='cuda:0'):
        """

        """
        super().__init__(net, opt, sched, logger, print_progress, device)

        self.loss_fn = MSLoss(*[PINNLoss_MSE(dh=5, dt=0.002, c=torch.Tensor(training_data.velocity_field_values)[None,None].to(device))])


    def forward_loss(self, data):
        """

        """
        input, label = data['wave_input'].transpose(2, 1) , data['wave_output'].transpose(2, 1)
        input = input.to(self.device)
        label = label.to(self.device)

        output = self.net(input)

        pinn = torch.cat((input,output),axis=1)

        loss = self.loss_fn(pinn=pinn)

        return loss[0], {'Loss':loss[0], 'Loss PINN MSE':loss[1]} #,'Loss MSE':loss[1],'Loss GDL':loss[2],'Loss MAE':loss[3]} # Elements in the dict : only for printing

# Create the model
model = AutoEncoderModel(net, opt=optimizer, sched=None, logger=None, print_progress=False, device=device)

# Train the model
model.train(epochs, train_loader, checkpoint_path=checkpoint_path, checkpoint_freq=5, save_best=None)

# Save
#model.save_best(export_path=save_dir + save_pt_best)
model.save(export_path=save_dir + save_pt)
model.save_outputs(export_path=save_dir + save_txt)