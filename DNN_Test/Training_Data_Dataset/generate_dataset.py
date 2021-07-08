import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

import h5py
import cv2
import os 

class Generate_Dataset():
    def __init__(self,path,event):
        """
        path : input path to get the salvus .h5 file
        event : event in the salvus output folder 

        hf_path = path + event + '/output.h5'
        """
        self.path = path
        self.event = event

        self.hf_path = self.path + self.event + '/output.h5'

        self.hf = h5py.File(self.hf_path, 'r')

    def save_figures(self,folder_name,start,end,offset=4):
        """
        start : first output wavefield to predict
        end : last output wavefield to predict
        offset : number of wavefield to predict the output
        """

        # Create output dir 
        if not os.path.exists(folder_name):
            os.mkdir(folder_name)

        # # # Store all the pictures

        # CSV to store figure
        df_figure = pd.DataFrame()
        df_figure['X'] = self.hf['coordinates_ACOUSTIC'][:,0][:,0].astype(float)
        df_figure['Y'] = self.hf['coordinates_ACOUSTIC'][:,0][:,1].astype(float)

        for fig_i in range(start-offset,end+1):

            df_figure['Pressure'] = self.hf['volume']['phi'][fig_i][:,0].mean(axis=1) # Add pressure

            df_show = df_figure.pivot_table(values='Pressure',index='Y',columns='X').sort_index(axis=0,ascending=False) # Pivot Table to reorder the data

            name_0 = f'{self.event}_{fig_i}.tiff'
            cv2.imwrite(folder_name + name_0,df_show.values) # Save figure

    def save_csv(self,file_name,start,end):
        # # # Create the dataset in CSV

        # CSV to store figure name
        csv_name = pd.DataFrame()

        for i,fig_i in enumerate(range(start,end+1)):
            name_0 = f'{self.event}_{fig_i}.tiff'

            csv_name.loc[i,'y'] = name_0
            csv_name.loc[i,'y_number'] = fig_i

        # Save csv with figure name
        csv_name.to_csv(file_name)

if __name__ == "__main__":

    path = f"../NoCrack/Moseley_EARTH/" #Event0000/output.h5"
    event = 'Event0000'

    gen_dataset = Generate_Dataset(path=path,event=event)

    folder_name = 'Moseley_EARTH/'
    start = 10
    end = 20

    gen_dataset.save_figures(folder_name=folder_name,start=start,end=end)

    file_name = 'Moseley_Earth_Event0000_Continuous.csv'

    gen_dataset.save_csv(file_name=file_name,start=start,end=end)
