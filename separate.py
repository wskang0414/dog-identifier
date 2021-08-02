import os
import shutil
import numpy as np
import pandas as pd

# Set data directory paths
data_dir = './data'


# Split train dataset into train and validation sets
def split_indices(n, val_pct, seed):
    n_val = int(val_pct * n)
    np.random.seed(seed)
    idxs = np.random.permutation(n)
    return idxs[n_val:], idxs[:n_val]


train_dir = os.path.join(data_dir, 'train')
valid_dir = os.path.join(data_dir, 'valid')

# 20% of train set is valid set
val_pct = 0.2
rand_seed = 42
num_train_files = os.listdir(train_dir)

train_indices, valid_indices = split_indices(
    len(num_train_files), val_pct, rand_seed)

# Read image file names from the csv file
train_labels = pd.read_csv('./labels.csv')
valid_path_list = []

for i in range(0, len(train_labels["id"])):
    seq = str(train_labels["id"][i]) + ".jpg"
    name = os.path.join(train_dir, seq)
    valid_path_list.append(name)

# Move the images for valid set to valid folder
for file in range(0, len(train_labels["id"])):
    shutil.move(valid_path_list[valid_indices[file]], valid_dir)
