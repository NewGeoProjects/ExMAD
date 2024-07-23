

## WARNING!! Before run step 4 please restart kernel!!


import os
import h5py
import numpy as np
import matplotlib.pyplot as plt
from tabulate import tabulate

# Directory 
image_dir = "./UNET/H5_raster/"
pred_dir = "./UNET/Output/"


##%%================= MULTI IMAGE (RGB+MASK+PREDICT) ===========================


# Iterate over each file in the pred_dir
for filename in os.listdir(image_dir):
    # print(filename)
    if filename.endswith(".h5"): 
        print(filename)
        
        # S2 RGB IMAGE PRE-EVENT
        image_path = os.path.join(image_dir, filename)
        with h5py.File(image_path, "r") as f:
            a_group_key = list(f.keys())[0]
            bands = f[a_group_key][()]  # returns as a numpy array
        b1 = np.transpose(bands[:,:,0])
        b2 = np.transpose(bands[:,:,1])
        b3 = np.transpose(bands[:,:,2])
        b4 = np.transpose(bands[:,:,3])
        rgb_pre = np.stack((b3, b2, b1), axis=-1)
        rgb_pre = rgb_pre / np.max(rgb_pre)
        

        # PREDICTION PRE-EVENT IMAGE
        pred_path = os.path.join(pred_dir, filename)
        with h5py.File(pred_path, "r") as f:
            a_group_key = list(f.keys())[0]
            pred_pre = f[a_group_key][()]  # returns as a numpy array
        
            
       
        ## PLOT figure
        fig, axes = plt.subplots(1, 2, figsize=(12, 4))
        axes[0].imshow(rgb_pre)
        axes[0].set_title(f"Image: {filename}, RGB")
        axes[0].axis('off')
        axes[1].imshow(pred_pre, cmap='viridis')
        axes[1].set_title(f"Image: {filename}, Prediction")
        axes[1].axis('off')
        plt.tight_layout()
        
        ## Save figure
        fig_dir = f"UNET/Output_fig/{filename}.png"
        plt.savefig(fig_dir)
        
        ## Show figure
        plt.show()
        
        
        