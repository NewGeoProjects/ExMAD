# -*- coding: utf-8 -*-
"""
Created on Mon Jul 22 17:11:34 2024

@author: Michele Licata
"""

import argparse
import numpy as np
import os
import torch
import torch.nn as nn
from torch.utils import data
import torch.backends.cudnn as cudnn
from utils.tools import *
from dataset.landslide_dataset import LandslideDataSet
from model.Networks import unet
import h5py

name_classes = ['Non-Landslide','Landslide']
epsilon = 1e-14

def importName(modulename, name):
    """ Import a named object from a module in the context of this function.
    """
    try:
        module = __import__(modulename, globals(), locals(  ), [name])
    except ImportError:
        return None
    return vars(module)[name]

def get_arguments():

    parser = argparse.ArgumentParser(description="Baseline method for Land4Seen")
    
    parser.add_argument("--data_dir", type=str, default='./H5_raster/',
                        help="dataset path.")
    parser.add_argument("--model_module", type =str, default='model.Networks',
                        help='model module to import')
    parser.add_argument("--model_name", type=str, default='unet',
                        help='modle name in given module')
    parser.add_argument("--valid_list", type=str, default='./dataset/image_list.txt',
                        help="test list file.")                
    parser.add_argument("--num_classes", type=int, default=2,
                        help="number of classes.")               
    parser.add_argument("--num_workers", type=int, default=0,
                        help="number of workers for multithread dataloading.")
    parser.add_argument("--gpu_id", type=int, default=0,
                        help="gpu id in the training.")
    parser.add_argument("--snapshot_dir", type=str, default='./Output/',
                        help="where to save predicted maps.")
    parser.add_argument("--restore_from", type=str, default='./Exp/ExMAD_unet_trained.pth',
                        help="trained model.")
    return parser.parse_args()


def main():
    args = get_arguments()
    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu_id)
    snapshot_dir = args.snapshot_dir
    if os.path.exists(snapshot_dir)==False:
        os.makedirs(snapshot_dir)
        
    cudnn.enabled = True
    cudnn.benchmark = True
    
    # Create network   
    model = unet(n_classes=args.num_classes)
   
    saved_state_dict = torch.load(args.restore_from)  
    model.load_state_dict(saved_state_dict)

    model = model.cuda()

    valid_loader = data.DataLoader(
                    LandslideDataSet(args.data_dir, args.valid_list, set='unlabeled'),
                    batch_size=1, shuffle=False, num_workers=args.num_workers, pin_memory=True)

    print('Computing....')
    model.eval()
   
    for index, batch in enumerate(valid_loader):  
        image, _, name = batch
        print(name)
        image = image.float().cuda()
        name = name[0]
        print(index+1, '/', len(valid_loader), ': Testing ', name)  
        print(image.shape)
        with torch.no_grad():
            pred = model(image)
        _, _, h, w = image.shape
        input_size = (w, h)
        interp = nn.Upsample(size=(input_size[1], input_size[0]), mode='bilinear')
        pred_interp = interp(pred) 
        _,predict_labels = torch.max(pred_interp, 1)
        predict_labels = predict_labels.squeeze().data.cpu().numpy().astype('uint8')   
        
        with h5py.File(snapshot_dir+name,'w') as hf:
            hf.create_dataset('mask', data=predict_labels)
 
if __name__ == '__main__':
    main()
