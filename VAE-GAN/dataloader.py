import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
import torchvision.utils as vutils
import pandas as pd
import pydicom
from torch.utils.data import DataLoader, Dataset
from PIL import Image
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

"""def dataloader(batch_size):
  dataroot="/content/drive/My Drive/celeba"
  transform=transforms.Compose([ transforms.Resize(64),transforms.CenterCrop(64),transforms.ToTensor(),transforms.Normalize((0.5),(0.5))])
  dataset=torchvision.datasets.MNIST(root=dataroot, train=True,transform=transform, download=True)
  data_loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True)
  return data_loader"""

class CustomDataset(Dataset):
    def __init__(self, dataframe, transform=None):
        self.dataframe = dataframe
        self.transform = transform

    def __len__(self):
        return len(self.dataframe)

    def __getitem__(self, idx):
        img_path = self.dataframe.iloc[idx, 0]
        dcm_data = pydicom.dcmread(img_path)

        image = dcm_data.pixel_array
        image = Image.fromarray(image)
        label = int(self.dataframe.iloc[idx, 1])

        if self.transform:
            image = self.transform(image)

        return image, label

def dataloader(batch_size):
  data_entry = pd.read_csv('RSNA.csv')
  data_entry.columns = ['full_path', 'Label']
  transform=transforms.Compose([ transforms.Resize(64),transforms.CenterCrop(64),transforms.ToTensor(),transforms.Normalize((0.5),(0.5))])
  dataset=CustomDataset(data_entry, transform=transform)
  data_loader = DataLoader(dataset, batch_size=64, shuffle=True)
  return data_loader
