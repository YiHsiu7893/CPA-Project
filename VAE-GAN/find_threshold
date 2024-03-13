import torch
from torch.autograd import Variable
torch.autograd.set_detect_anomaly(True)
from dataloader import dataloader
from models import VAE_GAN
import os
from tqdm import tqdm 
import csv

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

data_loader=dataloader(64)
gen=VAE_GAN().to(device)

weights_path =  'model_weight/generator/generator_weights_50.pth' 
if os.path.exists(weights_path):
    gen.load_state_dict(torch.load(weights_path))
    print("use pretrained weight") 

thresholds = [i for i in range(80, 500, 20)]
best_threshold = 0
best_f1 = 0


for threshold in thresholds:
    total = 0
    right = 0
    tp = 0
    fp = 0
    fn = 0
    for data,label in tqdm(data_loader, desc=f"testing"):
        datav = Variable(data).to(device)
        mean, logvar, rec_enc = gen(datav)
        reconstruct_error = torch.abs(rec_enc - datav)
        reconstruct_error = reconstruct_error.view(reconstruct_error.size(0), -1).sum(dim=1)
        
        result = torch.zeros_like(reconstruct_error)  
        result[reconstruct_error > threshold] = 1
        
        total += len(datav)
        right += (result == label).sum().item()
        tp += ((result == label) & (label == 1)).sum().item()
        fp += ((result != label) & (label == 0)).sum().item()
        fn += ((result != label) & (label == 1)).sum().item()
    
    accuracy = right / total
    precision = tp / (tp + fp + 1e-9)
    recall = tp / (tp + fn + 1e-9)
    f1 = 2 * (precision * recall) / (precision + recall + 1e-9)
    
    if f1 > best_f1:
        best_f1 = f1
        best_threshold = threshold
    print('threshold: %d, accuracy: %f, recall: %f, precision: %f, F1: %f' %(threshold,accuracy,recall,precision,f1))
    data = [threshold, accuracy, recall, precision, f1]
    with open('find_threshold.csv', mode='a', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(data)
        
print('accuracy %f' %(right/total))
