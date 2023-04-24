import os
import tarfile

# Set the source folder and destination folder paths
src_folder = '/mnt/disks/persist/datasets/improved_aesthetics_6plus_data/'
train_folder = '/mnt/disks/persist/datasets/vae_data/train/'
val_folder = '/mnt/disks/persist/datasets/vae_data/val/'

# Get a list of all files in the source folder
file_list = []

# Loop through all files in the directory
for filename in os.listdir(src_folder):
    # Check if the file has the .atr extension
    if filename.endswith('.tar'):
        # If we have found 10 files, break the loop
        if len(file_list) <= 12:
            # Append the file name to the list
            file_list.append(filename)
        else:
            break   
             
# Loop through the tar files in the folder
for filename in file_list[:10]:
    # Open the tar file and extract its contents into the destination folder
    with tarfile.open(src_folder + filename, 'r') as tar:
        # Loop through all files in the tar file
        for member in tar.getmembers():
            # Check if the member is a JPG file
            if member.name.endswith('.jpg'):
                # Extract the JPG file into the destination folder
                tar.extract(member, path=train_folder)

# Loop through the tar files in the folder
for filename in file_list[10:]:
    # Open the tar file and extract its contents into the destination folder
    with tarfile.open(src_folder + filename, 'r') as tar:
        # Loop through all files in the tar file
        for member in tar.getmembers():
            # Check if the member is a JPG file
            if member.name.endswith('.jpg'):
                # Extract the JPG file into the destination folder
                tar.extract(member, path=val_folder)