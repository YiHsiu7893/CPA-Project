import torch
import torchvision.transforms as transforms
import torchvision.models as models
from PIL import Image

import torch.nn as nn
from einops import rearrange

import os
import matplotlib.pyplot as plt
import numpy as np
from glob import glob
from torch.optim import Adam
from skimage import img_as_ubyte

from tqdm import tqdm 
from torchvision import transforms

import pandas as pd
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, Dataset

class CustomDataset(Dataset):
    def __init__(self, dataframe, transform=None):
        self.dataframe = dataframe
        self.transform = transform

    def __len__(self):
        return len(self.dataframe)

    def __getitem__(self, idx):
        img_name = self.dataframe.iloc[idx, 0]
        img_path = self.dataframe.iloc[idx, 2]
        image = Image.open(img_path).convert('RGB')
        label = int(self.dataframe.iloc[idx, 3])


        if self.transform:
            image = self.transform(image)

        return image, label

if __name__=="__main__":
    
    EPOCH = 3
    LR = 0.0003
    
    device = "cpu"
    #device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using device:", device)
    
    data_entry_1 = pd.read_csv('NF_Con_1vs3.csv')
    data_entry_2 = pd.read_csv('NF_Eff_1vs3.csv')
    data_entry_3 = pd.read_csv('NF_Mass_1vs3.csv')
    data_entry_1.columns = ['Image Index', 'Finding Labels']
    data_entry_2.columns = ['Image Index', 'Finding Labels']
    data_entry_3.columns = ['Image Index', 'Finding Labels']
    my_glob = glob('data/images*/images/*.png')
    all_image_paths = {os.path.basename(x): x for x in my_glob}
    data_entry_1['full_path'] = data_entry_1['Image Index'].map(all_image_paths.get)
    data_entry_2['full_path'] = data_entry_2['Image Index'].map(all_image_paths.get)
    data_entry_3['full_path'] = data_entry_3['Image Index'].map(all_image_paths.get)
    data_entry_1['label'] = data_entry_1['Finding Labels'].apply(lambda target: 1 if 'Consolidation' in target else 0)
    data_entry_2['label'] = data_entry_2['Finding Labels'].apply(lambda target: 1 if 'Effusion' in target else 0)
    data_entry_3['label'] = data_entry_3['Finding Labels'].apply(lambda target: 1 if 'Mass' in target else 0)
    print(data_entry_1)
    print(data_entry_2)
    print(data_entry_3)
    
    train_set_1, test_set_1 = train_test_split(data_entry_1, test_size = 0.2, random_state = 1999)
    train_set_2, test_set_2 = train_test_split(data_entry_2, test_size = 0.2, random_state = 1999)
    train_set_3, test_set_3 = train_test_split(data_entry_3, test_size = 0.2, random_state = 1999)
    print("train set:")
    print(train_set_1)
    
    transform = transforms.Compose([
        transforms.Resize(299),  # InceptionV3 模型的輸入大小為 299x299
        transforms.ToTensor(),
    ])
    
    train_data_1 = CustomDataset(train_set_1, transform=transform)
    train_data_2 = CustomDataset(train_set_2, transform=transform)
    train_data_3 = CustomDataset(train_set_3, transform=transform)
    test_data_1 = CustomDataset(test_set_1, transform=transform)
    test_data_2 = CustomDataset(test_set_2, transform=transform)
    test_data_3 = CustomDataset(test_set_3, transform=transform)

    train_loader_1 = DataLoader(train_data_1, batch_size=32, shuffle=True)
    train_loader_2 = DataLoader(train_data_2, batch_size=32, shuffle=True)
    train_loader_3 = DataLoader(train_data_3, batch_size=32, shuffle=True)
    test_loader_1 = DataLoader(test_data_1, batch_size=32, shuffle=False)
    test_loader_2 = DataLoader(test_data_2, batch_size=32, shuffle=False)
    test_loader_3 = DataLoader(test_data_3, batch_size=32, shuffle=False)
    
    model = models.inception_v3(pretrained=True)
    num_ftrs = model.fc.in_features
    model.fc1 = nn.Linear(num_ftrs, 2)
    model.fc2 = nn.Linear(num_ftrs, 2) 
    model.fc3 = nn.Linear(num_ftrs, 2)
    model.fc = nn.Identity()

     
    
    weights_folder = 'model_weight/multitask/test3'
    if not os.path.exists(weights_folder):
        os.makedirs(weights_folder)
        print("make dir")
        
    """weights_path =  'model_weight/multitask/test1/NIH_weights_0.pth'   
    if os.path.exists(weights_path):
        model.load_state_dict(torch.load(weights_path))
        print("use pretrained weight")"""
    
    
    criterion_1 = nn.CrossEntropyLoss()
    criterion_2 = nn.CrossEntropyLoss()
    criterion_3 = nn.CrossEntropyLoss()
    optimizer = Adam(model.parameters(), lr=LR)
    
    model.to(device)
    model.train()
    
    columns = ['Epoch', 'Accuracy']
    accuracy_df = pd.DataFrame(columns=columns)

    for epoch in range(EPOCH):  
        model.train()
        for batch_1, batch_2, batch_3  in tqdm(zip(train_loader_1, train_loader_2, train_loader_3), desc=f"Epoch {epoch + 1}/{EPOCH}", total=len(train_loader_1)):
            inputs_1, labels_1 = batch_1
            inputs_2, labels_2 = batch_2
            inputs_3, labels_3 = batch_3
            
            inputs_1, labels_1 = inputs_1.to(device), labels_1.to(device)
            inputs_2, labels_2 = inputs_2.to(device), labels_2.to(device)
            inputs_3, labels_3 = inputs_3.to(device), labels_3.to(device)
            
            optimizer.zero_grad()
            
            output_1 = model(inputs_1).logits
            output_2 = model(inputs_2).logits
            output_3 = model(inputs_3).logits
            #print(output.shape)
            outputs_1 = model.fc1(output_1)
            outputs_2 = model.fc2(output_2)
            outputs_3 = model.fc3(output_3)
            
            loss_1 = criterion_1(outputs_1, labels_1)  # 根據每個二元分類器的標籤計算損失
            loss_2 = criterion_2(outputs_2, labels_2)
            loss_3 = criterion_3(outputs_3, labels_3)
            
            loss = loss_1 + loss_2 + loss_3
            loss.backward()
            optimizer.step()
            
        weight_path = os.path.join(weights_folder, f'NIH_weights_{epoch + 1}.pth')
        torch.save(model.state_dict(), weight_path)
            
            
        model.eval()
        correct_1, correct_2, correct_3 = 0, 0, 0
        total = 0
        TP, TN, FP, FN = 0, 0, 0, 0
        with torch.no_grad():
            for batch_1, batch_2, batch_3  in tqdm(zip(test_loader_1, test_loader_2, test_loader_3), desc=f"Epoch {epoch + 1}/{EPOCH}", total=len(train_loader_1)):
                
                inputs_1, labels_1 = batch_1
                inputs_2, labels_2 = batch_2
                inputs_3, labels_3 = batch_3
                
                inputs_1, labels_1 = inputs_1.to(device), labels_1.to(device)
                inputs_2, labels_2 = inputs_2.to(device), labels_2.to(device)
                inputs_3, labels_3 = inputs_3.to(device), labels_3.to(device)
                #print(output.shape)
                
                output_1 = model(inputs_1)
                output_2 = model(inputs_2)
                output_3 = model(inputs_3)
                outputs_1 = model.fc1(output_1)
                outputs_2 = model.fc2(output_2)
                outputs_3 = model.fc3(output_3)
                _, predicted_1 = torch.max(outputs_1.data, 1)
                _, predicted_2 = torch.max(outputs_2.data, 1)
                _, predicted_3 = torch.max(outputs_3.data, 1)
                total += labels_1.size(0)
                correct_1 += (predicted_1 == labels_1).sum().item()
                correct_2 += (predicted_2 == labels_2).sum().item()
                correct_3 += (predicted_3 == labels_3).sum().item()
                """TN += torch.sum((predicted == 0) & (labels == 0)).detach().item()
                TP += torch.sum((predicted == 1) & (labels == 1)).detach().item()
                FN += torch.sum((predicted == 0) & (labels == 1)).detach().item()
                FP += torch.sum((predicted == 1) & (labels == 0)).detach().item()"""
        accuracy_1 = correct_1 / total
        accuracy_2 = correct_2 / total
        accuracy_3 = correct_3 / total
        #recall = TP/(TP+FN)
        print('Accuracy of the Effusion on the test images: %d %%' % (100 * accuracy_1))
        print('Accuracy of the Consolidation on the test images: %d %%' % (100 * accuracy_2))
        print('Accuracy of the Mass on the test images: %d %%' % (100 * accuracy_3))
        #print('Recall of the network on the test images: %d %%' % (100 * recall))
        """accuracy_df = accuracy_df.append({'Epoch': epoch + 1, 'Accuracy': accuracy}, ignore_index=True)
    accuracy_df.to_csv('acc.csv', index=False, mode='a',header=not os.path.exists('acc.csv'))
    
    model.eval()
    
    fp_folder = 'result/inception/fp'
    fn_folder = 'result/inception/fn'
    if not os.path.exists(fp_folder):
        os.makedirs(fp_folder)

    if not os.path.exists(fn_folder):
        os.makedirs(fn_folder)

    correct_1, correct_2, correct_3 = 0, 0, 0
    total = 0
    TP, TN, FP, FN = 0, 0, 0, 0
    j = 0
    with torch.no_grad():
        for inputs, labels_1, labels_2 ,labels_3 in tqdm(test_loader_1, desc="Testing"):
            inputs, labels_1, labels_2 ,labels_3 = inputs.to(device), labels_1.to(device), labels_2.to(device), labels_3.to(device)
            output = model(inputs)
                #print(output.shape)
            outputs_1 = model.fc1(output)
            outputs_2 = model.fc2(output)
            outputs_3 = model.fc3(output)
            _, predicted_1 = torch.max(outputs_1.data, 1)
            _, predicted_2 = torch.max(outputs_2.data, 1)
            _, predicted_3 = torch.max(outputs_3.data, 1)
            total += labels_1.size(0)
            correct_1 += (predicted_1 == labels_1).sum().item()
            correct_2 += (predicted_2 == labels_2).sum().item()
            correct_3 += (predicted_3 == labels_3).sum().item()
            TN += torch.sum((predicted == 0) & (labels == 0)).detach().item()
            TP += torch.sum((predicted == 1) & (labels == 1)).detach().item()
            FN += torch.sum((predicted == 0) & (labels == 1)).detach().item()
            FP += torch.sum((predicted == 1) & (labels == 0)).detach().item()
        accuracy_1 = correct_1 / total
        accuracy_2 = correct_2 / total
        accuracy_3 = correct_3 / total
        #recall = TP/(TP+FN)
        print('Accuracy of the Effusion on the test images: %d %%' % (100 * accuracy_1))
        print('Accuracy of the Consolidation on the test images: %d %%' % (100 * accuracy_2))
        print('Accuracy of the Mass on the test images: %d %%' % (100 * accuracy_3))

    #accuracy = correct / total
    #recall = TP/(TP+FN)
    #print('Accuracy of the network on the test images: %d %%' % (100 * accuracy))
    #print('Recall of the network on the test images: %d %%' % (100 * recall))
    """
