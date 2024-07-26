# -*- coding: utf-8 -*-
"""
Created on Mon Jul 22 17:11:34 2024

@author: Michele Licata
"""


import os
import rasterio
import h5py

# Directory paths
input_directory = './Input'
output_directory = './UNET/H5_raster'
output_list_file = './UNET/dataset/image_list.txt'

# Ensure the output directory exists
if not os.path.exists(output_directory):
    os.makedirs(output_directory)

# Get a list of all files in the input directory
file_list = os.listdir(input_directory)

# Filter the list to include only .tif files
tif_file_list = [filename for filename in file_list if filename.endswith(".tif")]

# List to store the HDF5 filenames
h5_filenames = []

for tif_filename in tif_file_list:
    print(tif_filename)
    # Remove the .tif extension and add the .h5 extension
    base_filename = os.path.splitext(tif_filename)[0]
    output_h5_filename = f"{base_filename}.h5"
    output_h5_path = os.path.join(output_directory, output_h5_filename)
    
    with rasterio.open(os.path.join(input_directory, tif_filename)) as src:
        raster = src.read()
        raster = raster.transpose()  # Transposing to match HDF5 storage conventions
        print(raster.shape)
        
        with h5py.File(output_h5_path, 'w') as hf:
            hf.create_dataset('img', data=raster)
    
    # Add the HDF5 filename to the list
    h5_filenames.append(output_h5_filename)

# Write the HDF5 filenames to the text file
with open(output_list_file, 'w') as txt_file:
    for h5_filename in h5_filenames:
        txt_file.write(h5_filename + "\n")

print("List of .h5 filenames exported to 'image_list.txt'")
