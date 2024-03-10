import torch.nn as nn
from einops import rearrange
from PIL import Image

import os
import shutil
from glob import glob
from torch.optim import Adam
from torch.nn import CrossEntropyLoss

from tqdm import tqdm 
from torchvision import transforms, datasets
from torchvision.transforms import Resize, ToTensor

import torchvision.transforms as transforms
import torch
import numpy as np
import cv2
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, Dataset, ConcatDataset, Subset

if __name__=="__main__":
    data_entry = pd.read_csv('rsna-pneumonia-detection-challenge/stage_2_detailed_class_info.csv')
    data_entry.columns = ['Id', 'class']
    data_entry['Id'] = data_entry['Id'].apply(lambda x: x + '.dcm')
    
    my_glob = glob('rsna-pneumonia-detection-challenge/stage_2_train_images/*.dcm')
    all_image_paths = {os.path.basename(x): x for x in my_glob}
    data_entry['full_path'] = data_entry['Id'].map(all_image_paths.get)
    data_entry = data_entry.dropna(subset=['full_path'])
    data_entry['label'] = data_entry['class'].apply(lambda target: 0 if target == 'Normal' else 1)
    data_entry = data_entry.drop(columns=['Id', 'class'])
    normal_dataset = data_entry[data_entry['label'] == 0]
    train_set, test_set = train_test_split(normal_dataset, test_size = 0.3, random_state = 1999)
    
    train_set.to_csv('training_data.csv', index=False)
    print(train_set)
