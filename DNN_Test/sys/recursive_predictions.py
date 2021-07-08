import time
import numpy as np
import pandas as pd
import torch

from skimage.metrics import structural_similarity as ssim

class RecursivePredictions():
    def __init__(self,net,dataset):
        """
        net : pytorch model create with the BaseModel class
        dataset : torch dataset
        """

        self.net = net
        self.dataset = dataset

        self.net = self.net.to('cuda:0')

        #initial_test_sample,number_of_timestep_to_predict,timestep_to_plot)
    
    def predict_all(self,initial_timestep,number_of_timestep_to_predict,time_timestep,
                        timestep_to_plot:list,x_locations :list ,y_locations :list,dh:int,model:str):
        """
        initial_timestep : (int) first timestep to start the predictions
        number_of_timestep_to_predict : (int) number of timestep to do the predictions
        time_timestep : (float) time between two time steps 
        timestep_to_plot : (list) the timesteps to plot
        x_locations : (list)
        y_locations : (list)
        dh : (int)
        """
        total_time = 0

        # Wavefield to start the predictions
        initial_test_sample = self.dataset.__getitem__(initial_timestep)
        test_x = initial_test_sample['wave_input'].transpose(1, 0)[None]
        test_x = test_x.to('cpu')

        preds = np.zeros((len(timestep_to_plot),300,300))
        trues = np.zeros((len(timestep_to_plot),300,300))
        
        im = 0

        # Metrics 
        ts_vec = []
        time_vec = []

        receivers_pred = []
        receivers_true = []

        ssim_vec = []
        rmse_vec = []
        relative_norm = []
        avg_abs_error_vec = []
        ratio_abs_median_error_abs_max_true = []

        for ts in range(number_of_timestep_to_predict):
            
            test_x = test_x.to('cuda:0')

            epoch_start_time = time.time()

            if model in ['unet','encoder']:
            # Prediciton for UNET and ENCODER
                test_gen = self.net(test_x).detach()

            if model in ['multiscale']:
                # Prediction for MULTISCALE
                input_4 = test_x[:,:,::4,::4]
                input_2 = test_x[:,:,::2,::2]
                input_1 = test_x
                test_gen = self.net(input_4,input_2,input_1).detach()

            pred = test_gen[0,0].cpu().numpy()

            epoch_end_time = time.time()
            total_epoch_time = np.round(epoch_end_time - epoch_start_time, 2)
            total_time += total_epoch_time
            print(f'\nTotal time for the timestep {ts} in seconds {total_epoch_time}')

            # Ground truth wavefield
            test_gt = self.dataset.__getitem__(initial_timestep+ts)['wave_output'].transpose(1, 0)[None].detach()
            true = test_gt[0,0].cpu().numpy()

            # # # Store selected wavefield
            if ts in timestep_to_plot:

                print(f"Saving wavefield {ts}")
                preds[im,:,:] = test_gen[0,0].cpu().numpy()
                trues[im,:,:] = test_gt[0,0].cpu().numpy()
                im += 1

            # # # Compute metrics
            # Timestep 
            ts_vec.append(ts)

            # Time
            time_vec.append(ts * time_timestep)

            # Save receivers values
            receivers_pred.append(pred[(x_locations/dh).astype(int),(y_locations/dh).astype(int)])
            receivers_true.append(true[(x_locations/dh).astype(int),(y_locations/dh).astype(int)])

            # RMSE
            rmse = np.sqrt( np.square( np.subtract( true , pred )).mean())
            rmse_vec.append(rmse)

            # SSIM
            ssim_vec.append(ssim(true, pred, data_range=true.max() - true.min()))
            
            # relative norm
            relative_norm.append(np.linalg.norm(np.subtract( true , pred ))/np.linalg.norm(true))

            # Average Absolute Error
            avg_abs_error = np.abs(np.subtract(true,pred)).mean()
            avg_abs_error_vec.append(avg_abs_error)

            # Ratio max abs error and max true value
            ratio_abs_median_error_abs_max_true.append( np.median(np.abs(pred-true)) / np.max(np.abs(true)) )

            # # # Update
            if self.dataset.velocity_field is not None:
                test_x = torch.cat((torch.cat((test_x[:,1:4,:,:],test_gen),dim=1),test_x[:,4:,:,:]),dim=1)
            else:
                test_x = torch.cat((test_x,test_gen),dim=1)[:,1:,:,:]

        # # # Dict Metrics 
        dict_metrics = {'ts':ts_vec,'time':time_vec,'receivers_pred':receivers_pred,'receivers_true':receivers_true,'rmse':rmse_vec,'ssim':ssim_vec,'relative_norm':relative_norm,'avg_abs_error':avg_abs_error_vec,'median_error_max_true':ratio_abs_median_error_abs_max_true,'total_time':total_time}

        # # # Errors - preds, trues and errors will be used to print the wavefields
        errors = preds - trues

        # # # Time print
        print(f'\nTotal time for the predictions [s] for {number_of_timestep_to_predict} timesteps :',np.round(total_time,2))

        return preds, trues, errors, dict_metrics 