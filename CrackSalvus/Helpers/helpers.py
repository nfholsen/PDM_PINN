from salvus.mesh.structured_grid_2D import StructuredGrid2D
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import cv2
import os


def get_mesh(nelem_x=25,nelem_y=25,max_x=1,max_y=1):
    mesh = StructuredGrid2D.rectangle(
    nelem_x=nelem_x, nelem_y=nelem_y, max_x=max_x, max_y=max_y
).get_unstructured_mesh()
    mesh.find_side_sets()
    return mesh

def plot_setup(x,src,recs):
    fig, ax = plt.subplots(figsize=(5,5))

    ax.plot([0,x,x,0,0],[0,0,x,x,0])

    # Plot receivers - in black below
    for i in range(len(recs)):
        ax.scatter(
        recs[i].get_dictionary()['location'][0],recs[i].get_dictionary()['location'][1], c='k'
    )
    # Plot sources - in red below
    for i in range(len(src)):
        ax.scatter(
        src[i].get_dictionary()['location'][0],src[i].get_dictionary()['location'][1], c ='r'
    )
    plt.show()

def plot_wavefield(hf,timestep,mesh_type='acoustic'):

    if mesh_type == 'acoustic':
        coordinates_type = 'coordinates_ACOUSTIC'
        wave_type = 'phi'
    elif mesh_type == 'elastic': 
        coordinates_type = 'coordinates_ELASTIC'
        wave_type = 'displacement'

    # # # Extract locations
    locations = hf[coordinates_type][:].reshape(hf[coordinates_type].shape[0]*hf[coordinates_type].shape[1],hf[coordinates_type].shape[2])

    # # # Create DF with locations reshaped
    df_loc = pd.DataFrame(locations,columns={'X','Y'})
    df_loc['Y'] = np.round(df_loc['Y'],10)
    df_loc['X'] = np.round(df_loc['X'],10)

    number_of_subplots = hf['volume'].get(wave_type).shape[2]

    print('number_of_subplots',number_of_subplots)

    # # # Extract values
    fig,ax = plt.subplots(1,number_of_subplots,figsize=(5,5),squeeze=False)
    ax = ax.ravel()

    for data in range(number_of_subplots):
        axis = ['X','Y']

        # # # Extract values
        values = hf['volume'].get(wave_type)[timestep][:,data,:hf[coordinates_type].shape[1]].reshape(hf[coordinates_type].shape[0]*hf[coordinates_type].shape[1],1)

        # # # Create DF with values reshaped
        df_loc[f'DISP_{axis[data]}'] = values

        # Pivot to have to optain numpy array 
        df_clean = df_loc.pivot_table(values=f'DISP_{axis[data]}', index='X', columns='Y')
        
        df_clean.sort_index(axis=0 ,ascending=False,inplace=True)
        
        ax[data].imshow(df_clean.values,extent=[0,1,0,1],vmin=df_clean.values.min(), vmax=df_clean.values.max())
    plt.show()

def create_video(path,hf,mesh_type):

    if not os.path.isdir(path + 'video/'):
        os.mkdir(path + 'video/')

    if mesh_type == 'acoustic':
        coordinates_type = 'coordinates_ACOUSTIC'
        wave_type = 'phi'
    elif mesh_type == 'elastic': 
        coordinates_type = 'coordinates_ELASTIC'
        wave_type = 'displacement'

    # # # Extract number of timestep
    number_of_ts = hf['volume'].get(wave_type).shape[0]

    # # # Extract locations
    locations = hf[coordinates_type][:].reshape(hf[coordinates_type].shape[0]*hf[coordinates_type].shape[1],hf[coordinates_type].shape[2])

    # # # Create DF with locations reshaped
    df_loc = pd.DataFrame(locations,columns={'X','Y'})
    df_loc['Y'] = np.round(df_loc['Y'],10)
    df_loc['X'] = np.round(df_loc['X'],10)

    number_of_subplots = hf['volume'].get(wave_type).shape[2]

    print('number_of_subplots',number_of_subplots)

    for img in range(number_of_ts):
        
        fig,ax = plt.subplots(1,number_of_subplots,figsize=(5,5),squeeze=False)
        ax = ax.ravel()

        for data in range(number_of_subplots):
            axis = ['X','Y']

            # # # Extract values
            values = hf['volume'].get(wave_type)[img][:,data,:hf[coordinates_type].shape[1]].reshape(hf[coordinates_type].shape[0]*hf[coordinates_type].shape[1],1)

            # # # Create DF with values reshaped
            df_loc[f'DISP_{axis[data]}'] = values

            # Pivot to have to optain numpy array 
            df_clean = df_loc.pivot_table(values=f'DISP_{axis[data]}', index='X', columns='Y')
            
            df_clean.sort_index(axis=0 ,ascending=False,inplace=True)
            
            ax[data].imshow(df_clean.values,extent=[0,1,0,1],vmin=df_clean.values.min(), vmax=df_clean.values.max())

        plt.savefig(path + f'video/ts_{img}.png')
        plt.close()

    width = 500
    height = 500
    channel = 3

    fps = 24

    fourcc = cv2.VideoWriter_fourcc(*'MP42')

    video = cv2.VideoWriter(path+'wave_propagation.avi', fourcc, float(fps), (width, height))

    directory = path+'video/'

    for frame in range(number_of_ts):
    
        img_name = f'ts_{frame}.png'
        img_path = os.path.join(directory, img_name)
        img = cv2.imread(img_path)
        img_resize = cv2.resize(img, (width, height))
     
        video.write(img_resize)
     
    video.release()