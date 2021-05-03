import numpy as np
import cv2
import pandas as pd
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader

class dataset(Dataset):
    def __init__(self, dir_path, csv_file):
        super(dataset, self).__init__()

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