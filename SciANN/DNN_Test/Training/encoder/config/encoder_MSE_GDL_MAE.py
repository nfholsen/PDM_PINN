import sys

sys.path.append('C:/Users/nils/Documents/PDM_PINN/SciANN/DNN_TEST/sys/')
#sys.path.append('C:/Users/nilso/Documents/EPFL/PDM/PDM_PINN/SciANN/DNN_TEST/sys/')

from loss import *
from encoder import AutoEncoder
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
batch_size = 4

# Data
data_dir = '../../../Training_Data/Moseley_Homogeneous/'
data_csv = '../../../Training_Data/Moseley_Homogeneous.csv'
event = 'Event0000'

# Paths
save_dir = '../results/'
save_pt_best = f'Best_L2_GDL_MAE_E{epochs}.pt'
save_pt = f'L2_GDL_MAE_E{epochs}.pt'
save_txt = f'L2_GDL_MAE_E{epochs}.yml'

checkpoint_path= f'checkpoint_L2_GDL_MAE_E{epochs}.pt'

# # # Data
training_data = dataset(data_dir,data_csv,event)
train_loader = DataLoader(training_data, batch_size=batch_size, shuffle=True)

net = AutoEncoder(in_channels=4,out_channels=1)

# Optimizer & Scheduler
optimizer = optim.Adam(net.parameters(), lr=0.0001, weight_decay=1e-6)

# Logger
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger()
try:
    logger.handlers[1].stream.close()
    logger.removeHandler(logger.handlers[1])
except IndexError:
    pass
logger.setLevel(logging.INFO)
file_handler = logging.FileHandler("log.txt")
file_handler.setLevel(logging.INFO)
file_handler.setFormatter(logging.Formatter('%(asctime)s | %(levelname)s | %(message)s'))
logger.addHandler(file_handler)

class AutoEncoderModel(BaseModel):
    def __init__(self, net, opt=None, sched=None, logger=None, print_progress=True, device='cuda:0'):
        """

        """
        super().__init__(net, opt, sched, logger, print_progress, device)

        self.loss_fn = MSLoss(*[nn.MSELoss(),GDLLoss(),nn.L1Loss()]) 

    def forward_loss(self, data):
        """

        """
        input, label = data['wave_input'].transpose(2, 1) , data['wave_output'].transpose(2, 1)
        input = input.to(self.device)
        label = label.to(self.device)

        output = self.net(input)

        loss = self.loss_fn(output, label)

        return loss[0], {'Loss':loss[0], 'Loss MSE':loss[1], 'Loss GDL':loss[2], 'Loss MAE':loss[3]} # Elements in the dict : only for printing

# Create the model
model = AutoEncoderModel(net, opt=optimizer, sched=None, logger=None, print_progress=False, device=device)

# Train the model
model.train(epochs, train_loader, checkpoint_path=checkpoint_path, checkpoint_freq=5, save_best=None)

# Save
#model.save_best(export_path=save_dir + save_pt_best)
model.save(export_path=save_dir + save_pt)
model.save_outputs(export_path=save_dir + save_txt)