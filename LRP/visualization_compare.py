import os

# Import saliency methods and models
from misc_functions import *

from explanation_generator import Baselines, LRP
from cifar10_LRP import ViT_cifar10
from torchvision.datasets.cifar import CIFAR10
from torch.utils.data import DataLoader

import torchvision.transforms as transforms
import torch
import numpy as np
import cv2
import matplotlib.pyplot as plt


def show_cam_on_image(img, mask):
    heatmap = cv2.applyColorMap(np.uint8(255 * mask), cv2.COLORMAP_JET)
    heatmap = np.float32(heatmap) / 255
    cam = heatmap + np.float32(img)
    cam = cam / np.max(cam)
    return cam

def generate_visualization_LRP(original_image, class_index=None):
    transformer_attribution = attribution_generator_LRP.generate_LRP(original_image.unsqueeze(0), method="transformer_attribution", index=class_index).detach()
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

def generate_visualization_attn_rollout(original_image, class_index=None):
    transformer_attribution = attribution_generator_attn_rollout.generate_rollout(original_image.unsqueeze(0)).detach()
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

def label_to_class_name(label):
    class_names = ["airplane", "automobile", "bird", "cat", "deer", "dog", "frog", "horse", "ship", "truck"]
    if 0 <= label < len(class_names):
        return class_names[label]
    else:
        return "Unknown"

if __name__ == "__main__":    
    # LRP
    model = ViT_cifar10(pretrained=True)
    model.eval()
    attribution_generator_LRP = LRP(model)
    attribution_generator_attn_rollout = Baselines(model)

    normalize = transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
    
    transform = transforms.Compose([
        transforms.Resize(32),
        normalize,
    ])
    
    result_folder = 'result/cifar10/compair'
    if not os.path.exists(result_folder):
        os.makedirs(result_folder)

    test_data = CIFAR10(root='./cifar10', train=False, download=False, transform=transforms.ToTensor(),)
    test_loader = DataLoader(test_data, shuffle=False, batch_size=32)

    correct_predictions = 0
    total_samples = 0
    with torch.no_grad():
        data_iter = iter(test_loader)
        inputs, labels = next(data_iter)
        outputs = model(inputs)
        print(outputs)
        _, predicted = torch.max(outputs, 1)
        print(predicted)
        total_samples += labels.size(0)
        correct_predictions += (predicted == labels).sum().item()
        
    for i in range(0,11):    
        img_tensor = torch.from_numpy(test_data.data[i])
        image = test_data[i][0]
        output = model(image.unsqueeze(0))
        right_class_name = label_to_class_name(test_data[i][1])
        predict_class_name = label_to_class_name(torch.argmax(output))
        visualization_LRP = generate_visualization_LRP(image)
        visualization_attn_rollout = generate_visualization_attn_rollout(image)
        
        plt.subplot(1, 3, 1)
        plt.imshow(img_tensor)
        plt.title(right_class_name)

        plt.subplot(1, 3, 2)
        plt.imshow(visualization_LRP)
        plt.title(predict_class_name)
        
        plt.subplot(1, 3, 3)
        plt.imshow(visualization_attn_rollout)
        plt.title(predict_class_name)

        file_name = f'{result_folder}/cifar10_{i}.png'
        plt.savefig(file_name)
        
        plt.show()
        
        

