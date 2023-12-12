import os
from tqdm import tqdm
import h5py

import argparse

# Import saliency methods and models
from misc_functions import *

from explanation_generator import Baselines, LRP
from cifar10_LRP import ViT_cifar10
from torchvision.datasets.cifar import CIFAR10
from torch.utils.data import DataLoader

import torchvision.transforms as transforms
from torchvision.transforms import ToPILImage, ToTensor
import torch
import numpy as np
import cv2
import matplotlib.pyplot as plt
import einops
from PIL import Image
import torchvision.transforms.functional as F
import matplotlib.pyplot as plt


def show_cam_on_image(img, mask):
    heatmap = cv2.applyColorMap(np.uint8(255 * mask), cv2.COLORMAP_JET)
    heatmap = np.float32(heatmap) / 255
    cam = heatmap + np.float32(img)
    cam = cam / np.max(cam)
    return cam

def generate_visualization(original_image, class_index=None):
    transformer_attribution = attribution_generator.generate_LRP(original_image.unsqueeze(0), method="transformer_attribution", index=class_index).detach()
    transformer_attribution = transformer_attribution.reshape(1, 1, 8, 8)
    transformer_attribution = torch.nn.functional.interpolate(transformer_attribution, scale_factor=4, mode='bilinear')
    transformer_attribution = transformer_attribution.reshape(32, 32).data.cpu().numpy()
    transformer_attribution = (transformer_attribution - transformer_attribution.min()) / (transformer_attribution.max() - transformer_attribution.min())

    image_transformer_attribution = original_image.permute(1, 2, 0).data.cpu().numpy()
    image_transformer_attribution = (image_transformer_attribution - image_transformer_attribution.min()) / (image_transformer_attribution.max() - image_transformer_attribution.min())
    vis = show_cam_on_image(image_transformer_attribution, transformer_attribution)
    vis =  np.uint8(255 * vis)
    vis = cv2.cvtColor(np.array(vis), cv2.COLOR_RGB2BGR)
    return vis

if __name__ == "__main__":    
    # LRP
    model = ViT_cifar10(pretrained=True)
    model.eval()
    attribution_generator = LRP(model)

    normalize = transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
    transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        normalize,
    ])

    # Dataset loader for sample images
    
    transform = transforms.Compose([
        transforms.Resize(32),
        transforms.ToTensor(),
        normalize,
    ])

    test_data = CIFAR10(root='./cifar10', train=False, download=False, transform=transform)
    test_loader = DataLoader(test_data, shuffle=False, batch_size=32)
    
    img_tensor = torch.from_numpy(test_data.data[0])
    print(img_tensor.shape)
    patches = img_tensor.permute(2, 0, 1)
    

    image_path = 'image/cifar10_1.png'
    image = Image.open(image_path)
    cat_image = transform(image)
    fig, axs = plt.subplots(1, 3)
    axs[0].imshow(image);
    axs[0].axis('off');
    output = model(cat_image.unsqueeze(0))
    print(output)
    #transforms.functional.to_pil_image(image).save(save_path)
    #output = model(img_tensor.unsqueeze(0))
    visulization = generate_visualization(cat_image)
    print(visulization.shape)
    plt.imshow(visulization)
    plt.show()

