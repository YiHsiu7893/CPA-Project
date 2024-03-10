import torch
from torch.autograd import Variable
torch.autograd.set_detect_anomaly(True)
from dataloader import dataloader
from models import VAE_GAN
import os
from tqdm import tqdm 

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

data_loader=dataloader(64)
gen=VAE_GAN().to(device)

weights_path =  'model_weight/generator/generator_weights_1.pth' 
if os.path.exists(weights_path):
    gen.load_state_dict(torch.load(weights_path))
    print("use pretrained weight") 

threshold = 700

total = 0
right = 0

for data,label in tqdm(data_loader, desc=f"testing"):
    datav = Variable(data).to(device)
    mean, logvar, rec_enc = gen(datav)
    reconstruct_error = torch.abs(rec_enc - datav)
    reconstruct_error = reconstruct_error.view(reconstruct_error.size(0), -1).sum(dim=1)
    
    result = torch.zeros_like(reconstruct_error)  
    result[reconstruct_error > threshold] = 1
    
    total += len(datav)
    right += (result == label).sum().item()

print('accuracy %f' %(right/total))
