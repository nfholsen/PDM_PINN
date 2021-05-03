from abc import ABC, abstractmethod
from typing import List, Tuple, Dict, Union
import logging
from logging import Logger
import yaml
import time
from datetime import timedelta

import torch
from torch.optim.lr_scheduler import _LRScheduler
from torch.optim.optimizer import Optimizer
from torch.utils.data import DataLoader
from torch import Tensor
import torch.nn as nn

import copy # To save the best model


class BaseModel(ABC):
    def __init__(self, net: nn.Module, opt: Optimizer = None, sched: _LRScheduler = None,
                 logger: Logger = None, print_progress: bool = True, device: str = 'cuda:0', **kwargs):
        """
        Abtract class defining a moodel based on Pytorch. It allows to save/load the model and train/evaluate it.
        Classes inheriting from the BaseModel needs to be initialized with a nn.Modules. This network can be trained using
        the passed optimizer/lr_scheduler with the self.train() methods. To be used, the children class must define two
        abstract methods:
            1. `forward_loss(data: Tuple[Tensor])` : define the processing of 1 batch provided by the DataLoader. `data`
               is the tuple of tensors given by the DataLoader. This method should thus define how the data is i) unpacked
               ii) how the forward pass with self.net is done iii) and how the loss is computed. The method should then
               return the loss.
            2. `validate(loader: DataLoader)` : define how the model is validated at each epoch. It takes a DataLoader
               for the validation data as input and should return a dictionnary of properties to print in the epoch
               summary (as {property_name : str_property_value}). No validation is performed if no valid_loader is passed
               to self.train()

        Note: the BaseModel has a dictionnary as attributes (self.outputs) that allow to store some values (training time,
              validation scores, epoch evolution, etc). This dictionnary can be saved as a YAML file using the save_outputs
              method. Any other values can be added to the self.outputs using self.outputs["key"] = value.

              If Logger is None, the outputs are displayed using `print`.
        """
        self.net = net
        self.best_net = net
        self.optimizer = opt
        self.lr_scheduler = sched
        self.logger = logger

        # where to print info
        self.print_fn = logger.info if logger else print

        self.device = device
        self.print_progress = print_progress

        self.outputs = {} # placeholder for any output to be saved in JSON

    def train(self, n_epochs: int, train_loader: DataLoader, valid_loader: DataLoader = None,
              checkpoint_path: str = None, checkpoint_freq: int = 10, save_best: str = None): # TODO : add best model (save best)
        """
        Train the self.net using the optimizer and scheduler using the data provided by the train_loader. At each epoch,
        the model can be validated using the valid_loader (if a valid loader is provided, the method self.validate must
        be implemented in the children). The model and training state is loaded/saved in a .pt file if checkpoint_path
        is provided. The model is then saved every checkpoint_freq epoch.
        """
        assert self.optimizer is not None, "An optimizer must be provided to train the model."

        # Load checkpoint if any
        try:
            checkpoint = torch.load(checkpoint_path, map_location=self.device)
            n_epoch_finished = checkpoint['n_epoch_finished']

            self.net.load_state_dict(checkpoint['net_state'])
            self.net = self.net.to(self.device) #  Redundant but required to avoid cpu cuda errors when loading checkpoint see : https://stackoverflow.com/questions/62136244/pytorch-device-problemcpu-gpu-when-load-state-dict-for-optimizer
            self.optimizer.load_state_dict(checkpoint['optimizer_state'])

            if save_best :
                best_metric = checkpoint['best_metric'] # TODO : add best model --> convert to tensor and load float instead
                self.best_net.load_state_dict(checkpoint['best_net_state']) # TODO : add best model 

            if self.lr_scheduler:
                self.lr_scheduler.load_state_dict(checkpoint['lr_state'])
            epoch_loss_list = checkpoint['loss_evolution']
            self.print_fn(f'Resuming from Checkpoint with {n_epoch_finished} epoch finished.')
        except FileNotFoundError:
            self.print_fn('No Checkpoint found. Training from beginning.')
            n_epoch_finished = 0
            epoch_loss_list = [] # Placeholder for epoch evolution

        self.net = self.net.to(self.device)
        
        # Train Loop
        for epoch in range(n_epoch_finished, n_epochs):
            self.net.train()
            epoch_start_time = time.time()

            n_batches = len(train_loader)

            for b, data in enumerate(train_loader):
                # Gradient descent step
                self.optimizer.zero_grad()
                loss = self.forward_loss(data)

                if type(loss) is tuple:
                    loss, loss_outputs = loss

                    if b == 0:
                        train_outputs = {name : 0.0 for name in loss_outputs.keys()}

                    train_outputs = {name : (value + loss_outputs[name].item() if type(loss_outputs[name]) is torch.Tensor else value + loss_outputs[name]) for name, value in train_outputs.items()}
                else: 
                    if b == 0: 
                        train_outputs = {'Loss' : 0.0}

                    train_outputs["Loss"] += loss.item()

                loss.backward()
                self.optimizer.step()

                if self.print_progress:
                    self.print_progessbar(b, n_batches, name='Train Batch', size=100, erase=True)

            # validate epoch
            if valid_loader:
                self.net.eval()
                valid_outputs = self.validate(valid_loader)
            else:
                valid_outputs = {}
            

            # print epoch stat
            self.print_fn(f"Epoch {epoch+1:04}/{n_epochs:04} | "
                          f"Time {timedelta(seconds=time.time()-epoch_start_time)} | "
                          + "".join([f"{name} {loss_i / n_batches:.5f} | " for name, loss_i in train_outputs.items()])
                          + "".join([f"{name} {val} | " for name, val in valid_outputs.items()])
                          )


            train_loss = {name : loss/n_batches for name, loss in train_outputs.items()}
            val_loss = {name : loss/n_batches for name, loss in valid_outputs.items()}

            epoch_loss_list.append([epoch+1, train_loss, val_loss])  # <-- TO CHECK

            # scheduler steps
            if self.lr_scheduler:
                self.lr_scheduler.step()

            # Update best model : TODO
            if save_best in list(train_outputs.keys()):
                if epoch + 1 == 1:
                    best_metric = train_outputs[save_best]
                    self.best_net = copy.deepcopy(self.net)
                
                if train_outputs[save_best] < best_metric:
                    best_metric = train_outputs[save_best]
                    self.best_best_net = copy.deepcopy(self.net)
            else:
                if epoch + 1 == 1:
                    self.print_fn('No matching metrics found. Tried : {}, Got : {}'.format(list(train_outputs.keys()),[str(save_best)]))

            # Save checkpoint
            if (epoch+1) % checkpoint_freq == 0 and checkpoint_path is not None:
                checkpoint = {
                    'n_epoch_finished': epoch+1,
                    'net_state': self.net.state_dict(),
                    'optimizer_state': self.optimizer.state_dict(),
                    'loss_evolution': epoch_loss_list
                }

                if save_best in list(train_outputs.keys()):
                    checkpoint['best_net_state'] = self.best_net.state_dict()
                    checkpoint['best_metric'] = best_metric

                if self.lr_scheduler:
                    checkpoint['lr_state'] = self.lr_scheduler.state_dict()
                torch.save(checkpoint, checkpoint_path)
                self.print_fn('\tCheckpoint saved.')

        self.outputs['train_evolution'] = epoch_loss_list

    def validate(self, loader: DataLoader) -> Dict:
        """
        --> Define how to validate the model
        output : Dictionnary {"Name" : str(Value)} e.g. {"Accuracy" : "87,5%", "Loss" : "0.03421", ...}
        """
        raise NotImplementedError("self.validate(loader) must be implemented when a valid Dataloader is passed to self.train().")

    @abstractmethod
    def forward_loss(self, data: Tuple[Tensor]) -> Union[Tensor,Tuple[Tensor,Dict]]:
        """
        --> Define Forward + Loss Computation from data provided by loader

        e.g.
        inputs, labels = data
        inputs = inputs.to(self.device)
        labels = labels.to(self.device)
        outputs = self.net(inputs)
        return loss_fn(outputs, labels)
        """
        pass

    def load(self, import_path: str, map_location: str = 'cuda:0'):
        """
        Load the model state dictionnary at the import path on the device specified by map_location.
        """
        loaded_state_dict = torch.load(import_path, map_location=map_location)
        self.net.load_state_dict(loaded_state_dict)

    def save(self, export_path: str):
        """
        Save model state dictionnary at the export_path on the device specified by map_location.
        """
        torch.save(self.net.state_dict(), export_path)

    def save_best(self, export_path: str):
        """
        Save best model state dictionnary at the export_path on the device specified by map_location.
        """
        torch.save(self.best_net.state_dict(), export_path)

    def save_outputs(self, export_path: str):
        """
        Save the output attribute dictionnary as a .yml specified by export_path.
        """
        with open(export_path, "w") as f:
            yaml.dump(self.outputs, f)

    def transfer_weight(self, import_path: str, map_location: str = 'cuda:0', verbose: bool = True):
        """
        Transfer all matching keys of the model state dictionnary at the import path to self.net.
        """
        # load pretrain weights
        init_state_dict = torch.load(import_path, map_location=map_location)
        # get self.net state dict
        net_state_dict = self.net.state_dict()
        # get common keys
        to_transfer_keys = {k:w for k, w in init_state_dict.items() if k in net_state_dict}
        if verbose:
            self.print_fn(f'{len(to_transfer_keys)} matching weight keys found on {len(init_state_dict)} to be tranferred to the net ({len(net_state_dict)} weight keys).')
        # update U-Net weights
        net_state_dict.update(to_transfer_keys)
        self.net.load_state_dict(net_state_dict)

    @staticmethod
    def print_progessbar(n: int, max: int, name: str = '', size: int = 10, end_char: str = '', erase: bool = False):
        """
        Print a progress bar. To be used in a for-loop and called at each iteration
        with the iteration number and the max number of iteration.
        ------------
        INPUT
            |---- n (int) the iteration current number
            |---- max (int) the total number of iteration
            |---- name (str) an optional name for the progress bar
            |---- size (int) the size of the progress bar
            |---- end_char (str) the print end parameter to used in the end of the
            |                    progress bar (default is '')
            |---- erase (bool) whether to erase the progress bar when 100% is reached.
        OUTPUT
            |---- None
        """
        print(f'{name} {n+1:04d}/{max:04d}'.ljust(len(name) + 12) \
            + f'|{"█"*int(size*(n+1)/max)}'.ljust(size+1) + f'| {(n+1)/max:.1%}'.ljust(6), \
            end='\r')

        if n+1 == max:
            if erase:
                print(' '.ljust(len(name) + size + 40), end='\r')
            else:
                print('')
