import os

# Import saliency methods and models
from misc_functions import *

from explanation_generator import Baselines, LRP
from NIH_Chest_X_rays import ViT_NIH
from torch.utils.data import DataLoader

from glob import glob
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, Dataset
import torch
import numpy as np
import cv2
import matplotlib.pyplot as plt
from PIL import Image
import pandas as pd
from sklearn.model_selection import train_test_split


def show_cam_on_image(img, mask):
    heatmap = cv2.applyColorMap(np.uint8(255 * mask), cv2.COLORMAP_JET)
    heatmap = np.float32(heatmap) / 255
    cam = heatmap + np.float32(img)
    cam = cam / np.max(cam)
    return cam

def generate_visualization_LRP(original_image, class_index=None):
    transformer_attribution = attribution_generator_LRP.generate_LRP(original_image.unsqueeze(0), method="transformer_attribution", index=class_index).detach()
    transformer_attribution = transformer_attribution.reshape(1, 1, 28, 28)
    transformer_attribution = torch.nn.functional.interpolate(transformer_attribution, scale_factor=8, mode='bilinear')
    transformer_attribution = transformer_attribution.reshape(224, 224).data.cpu().numpy()
    transformer_attribution = (transformer_attribution - transformer_attribution.min()) / (transformer_attribution.max() - transformer_attribution.min())

    image_transformer_attribution = original_image.permute(1, 2, 0).data.cpu().numpy()
    image_transformer_attribution = (image_transformer_attribution - image_transformer_attribution.min()) / (image_transformer_attribution.max() - image_transformer_attribution.min())
    vis = show_cam_on_image(image_transformer_attribution, transformer_attribution)
    vis =  np.uint8(255 * vis)
    vis = cv2.cvtColor(np.array(vis), cv2.COLOR_RGB2BGR)
    return vis

def generate_visualization_attn_rollout(original_image, class_index=None):
    transformer_attribution = attribution_generator_attn_rollout.generate_rollout(original_image.unsqueeze(0)).detach()
    transformer_attribution = transformer_attribution.reshape(1, 1, 28, 28)
    transformer_attribution = torch.nn.functional.interpolate(transformer_attribution, scale_factor=8, mode='bilinear')
    transformer_attribution = transformer_attribution.reshape(224, 224).data.cpu().numpy()
    transformer_attribution = (transformer_attribution - transformer_attribution.min()) / (transformer_attribution.max() - transformer_attribution.min())

    image_transformer_attribution = original_image.permute(1, 2, 0).data.cpu().numpy()
    image_transformer_attribution = (image_transformer_attribution - image_transformer_attribution.min()) / (image_transformer_attribution.max() - image_transformer_attribution.min())
    vis = show_cam_on_image(image_transformer_attribution, transformer_attribution)
    vis =  np.uint8(255 * vis)
    vis = cv2.cvtColor(np.array(vis), cv2.COLOR_RGB2BGR)
    return vis

def label_to_class_name(label):
    class_names = ["normal", "abnormal"]
    if 0 <= label < len(class_names):
        return class_names[label]
    else:
        return "Unknown"
    
class CustomDataset(Dataset):
    def __init__(self, dataframe, transform=None):
        self.dataframe = dataframe
        self.transform = transform

    def __len__(self):
        return len(self.dataframe)

    def __getitem__(self, idx):
        img_path = self.dataframe.iloc[idx]['full_path']
        #print(img_path)
        image = Image.open(img_path).convert('L')

        if self.transform:
            image = self.transform(image)

        label = self.dataframe.iloc[idx]['label'].item()
        #print(label)
        return image, label

if __name__ == "__main__":    
    # LRP
    model = ViT_NIH(pretrained=True)
    model.eval()
    attribution_generator_LRP = LRP(model)
    attribution_generator_attn_rollout = Baselines(model)

    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
    ])
    
    data_entry = pd.read_csv('Data_Entry_2017_v2020.csv')
    data_entry.columns = ['Image Index', 'Finding Labels', 'Follow_Up_#', 'Patient_ID', 'Patient_Age', 'Patient_Gender',
                'View_Position', 'Original_Image_Width', 'Original_Image_Height', 
                'Original_Image_Pixel_Spacing_X', 'Original_Image_Pixel_Spacing_Y']
    
    data_entry.drop(['Original_Image_Pixel_Spacing_X', 'Original_Image_Pixel_Spacing_Y',
                 'Original_Image_Width', 'Original_Image_Height','View_Position',
                'Follow_Up_#', 'Patient_ID', 'Patient_Age', 'Patient_Gender'], axis = 1, inplace = True)

    my_glob = glob('images*/images/*.png')
    all_image_paths = {os.path.basename(x): x for x in my_glob}
    data_entry['full_path'] = data_entry['Image Index'].map(all_image_paths.get)
    data_entry['label'] = data_entry['Finding Labels'].apply(lambda target: 0 if 'No Finding' in target else 1)
    print(data_entry)
    
    
    train_set, test_set = train_test_split(data_entry, test_size = 0.2, random_state = 1999)
    print("test set:")
    print(test_set)
    train_set.to_csv('train_set.csv', index=True, mode = 'w')
    
    test_data = CustomDataset(test_set, transform=transform)
    
    BATCH_SIZE = 32
    test_loader = DataLoader(test_data, shuffle=False, batch_size=BATCH_SIZE)

    """correct_predictions = 0
    total_samples = 0
    with torch.no_grad():
        data_iter = iter(test_loader)
        inputs, labels = next(data_iter)
        outputs = model(inputs)
        print(outputs)
        _, predicted = torch.max(outputs, 1)
        print(predicted)
        total_samples += labels.size(0)
        correct_predictions += (predicted == labels).sum().item()"""
        
    """img_tensor = torch.from_numpy(test_data.data[3])
    image = test_data[3][0]
    output = model(image.unsqueeze(0))
    print(output)
    right_class_name = label_to_class_name(test_data[3][1])
    predict_class_name = label_to_class_name(torch.argmax(output))
    visualization_LRP = generate_visualization_LRP(image)
    visualization_attn_rollout = generate_visualization_attn_rollout(image)
        
    plt.subplot(1, 3, 1)
    plt.imshow(img_tensor)
    plt.title(right_class_name)

    plt.subplot(1, 3, 2)
    plt.imshow(visualization_LRP)
    plt.title("LRP : " + predict_class_name)
        
    plt.subplot(1, 3, 3)
    plt.imshow(visualization_attn_rollout)
    plt.title("attn rollout : " + predict_class_name)

    plt.show() """   
    
    #test_data = [(np.array(data), label) for data, label in test_data]
    
    result_folder = 'result/NIH/compair/learningRate305'
    os.makedirs(result_folder, exist_ok=True)
    for i in range(0,20):   
        print(i) 
        img_tensor = torch.from_numpy(np.array(test_data[i][0]))
        image = test_data[i][0]
        output = model(image.unsqueeze(0))
        right_class_name = label_to_class_name(test_data[i][1])
        predict_class_name = label_to_class_name(torch.argmax(output))
        visualization_LRP = generate_visualization_LRP(image)
        visualization_attn_rollout = generate_visualization_attn_rollout(image)
        
        plt.subplot(1, 3, 1)
        plt.imshow(img_tensor.squeeze().numpy(), cmap='gray')
        plt.title(right_class_name)

        plt.subplot(1, 3, 2)
        plt.imshow(visualization_LRP)
        plt.title("LRP : " + predict_class_name)
        
        plt.subplot(1, 3, 3)
        plt.imshow(visualization_attn_rollout)
        plt.title("attn rollout : " + predict_class_name)

        file_name = f'{result_folder}/NIH_{i}.png'
        plt.savefig(file_name)
        
        plt.show()
        
        

