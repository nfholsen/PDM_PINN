import numpy as np
import cv2
import pandas as pd
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader

class dataset(Dataset):
    def __init__(self, dir_path, csv_file, velocity_field=None):
        super(dataset, self).__init__()

        self.dir_path = dir_path
        self.csv_file = pd.read_csv(csv_file,index_col=0)

        self.velocity_field = velocity_field # Path to velocity field

        if self.velocity_field:
            self.velocity_field_values = np.load(self.velocity_field)

        self.output = self.csv_file['y'] # Update : Changed y_number to y

        self.transform = transforms.Compose([
    transforms.ToTensor()
])

    def __len__(self):
        return self.csv_file.shape[0]
        
    def __getitem__(self, index):

        # Get the indice for the output wavefield
        self.output_im = self.csv_file.loc[index,'y_number'].astype(int)

        inputs = [cv2.imread(self.dir_path + f"{self.output[index].split('_')[0]}_{im}.tiff" ,cv2.IMREAD_UNCHANGED).T for im in range(self.output_im-4,self.output_im)]
        #print(list(self.dir_path + f"{self.output[index].split('_')[0]}_{im}.tiff" for im in range(self.output_im-4,self.output_im))) # For debugging

        if self.velocity_field:
            inputs.append(self.velocity_field_values.T)
        
        outputs = [cv2.imread(self.dir_path + f"{self.output[index].split('_')[0]}_{im}.tiff",cv2.IMREAD_UNCHANGED).T for im in [self.output_im]]
        #print(list(self.dir_path + f'{self.output[index]}' for im in [index])) # For debugging

        inputs = self.transform(np.array(inputs)).float()
        outputs = self.transform(np.array(outputs)).float()
        sample = {"wave_input": inputs,
                    "wave_input_label":self.output[index],
                    "wave_output": outputs,
                    "wave_output_label":self.output_im}
        return sample