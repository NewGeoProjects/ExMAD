import numpy as np
import torch
from torch.utils import data
from torch.utils.data import DataLoader
import h5py

class LandslideDataSet(data.Dataset):
    def __init__(self, data_dir, list_path, max_iters=None,set='label'):
        self.list_path = list_path
        self.set = set
        self.img_ids = [i_id.strip() for i_id in open(list_path)]
        
        if not max_iters==None:
            n_repeat = int(np.ceil(max_iters / len(self.img_ids)))
            self.img_ids = self.img_ids * n_repeat + self.img_ids[:max_iters-n_repeat*len(self.img_ids)]

        self.files = []

        if set=='labeled':
            for name in self.img_ids:
                img_file = data_dir + name
                label_file = data_dir + name.replace('img','mask').replace('image','mask')
                self.files.append({
                    'img': img_file,
                    'label': label_file,
                    'name': name
                })
        elif set=='unlabeled':
            for name in self.img_ids:
                img_file = data_dir + name
                self.files.append({
                    'img': img_file,
                    'name': name
                })
            
    def __len__(self):
        return len(self.files)


    def __getitem__(self, index):
        datafiles = self.files[index]
        
        if self.set=='labeled':
            with h5py.File(datafiles['img'], 'r') as hf:
                image = hf['img'][:]
            with h5py.File(datafiles['label'], 'r') as hf:
                label = hf['mask'][:]
            name = datafiles['name']
            image = np.asarray(image, np.float32)
            label = np.asarray(label, np.float32)
            image = image.transpose((-1, 0, 1))
            size = image.shape
           
            # Compute the mean and standard deviation for each channel
            channel_means = np.mean(image, axis=(1, 2))
            channel_std = np.std(image, axis=(1, 2))
            # Normalize each channel of the image
            normalized_image = np.zeros_like(image)
            for i in range(len(channel_means)):
                print(i, channel_means[i], channel_std)
                normalized_image[i, :, :] = (image[i, :, :] - channel_means[i]) / channel_std[i]
            image = normalized_image

            return image.copy(), label.copy(), np.array(size), name

        else:
            with h5py.File(datafiles['img'], 'r') as hf:
                image = hf['img'][:]
            name = datafiles['name']
                
            image = np.asarray(image, np.float32)
            image = image.squeeze()
            image = image.transpose()
            size = image.shape
           
            # Compute the mean and standard deviation for each channel
            channel_means = np.mean(image, axis=(1, 2))
            channel_std = np.std(image, axis=(1, 2))
            # Normalize each channel of the image
            normalized_image = np.zeros_like(image)
            for i in range(len(channel_means)):
                normalized_image[i, :, :] = (image[i, :, :] - channel_means[i]) / channel_std[i]
            image = normalized_image

            return image.copy(), np.array(size), name

 