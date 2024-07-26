# -*- coding: utf-8 -*-
"""
Created on Mon Jul 22 17:11:34 2024

@author: Michele Licata
"""

import os

# Directory path
directory = "./UNET/H5_raster"

# Get a list of all files in the directory
file_list = os.listdir(directory)

# Filter the list to include only .h5 files
h5_file_list = [filename for filename in file_list if filename.endswith(".h5")]

# Create a text file to store the list
with open("./UNET/dataset/image_list.txt", "w") as txt_file:
    for h5_filename in h5_file_list:
        h5_file_path = os.path.join(directory, h5_filename).replace("\\", "/")
        # Extract just the filename without the directory path
        filename = os.path.basename(h5_file_path)
        txt_file.write(filename + "\n")

print("List of .h5 filenames exported to 'image_list.txt'")
