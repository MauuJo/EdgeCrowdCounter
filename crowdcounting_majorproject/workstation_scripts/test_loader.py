import torch
import matplotlib.pyplot as plt
import numpy as np
from torch.utils.data import DataLoader
from dataset import CrowdDataset

# ⚠️ UPDATE THESE PATHS TO YOUR WORKSTATION FOLDERS
WORKSTATION_IMG_DIR = "D:/Path/To/UCF-QNRF/Train" 
WORKSTATION_H5_DIR  = "D:/Path/To/UCF-QNRF/Train_h5"

def test_dataloader():
    dataset = CrowdDataset(img_dir=WORKSTATION_IMG_DIR, h5_dir=WORKSTATION_H5_DIR)
    loader = DataLoader(dataset, batch_size=2, shuffle=True)
    
    images, heatmaps = next(iter(loader))
    
    # Un-normalize image for viewing
    img = images[0].numpy().transpose(1, 2, 0)
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    img = std * img + mean
    img = np.clip(img, 0, 1)
    
    heatmap = heatmaps[0].squeeze().numpy()
    
    plt.figure(figsize=(12, 6))
    plt.subplot(1, 2, 1)
    plt.imshow(img)
    plt.title("Random 512x512 Crop")
    
    plt.subplot(1, 2, 2)
    plt.imshow(heatmap, cmap='jet')
    plt.title(f"Target 64x64 Heatmap (Count: {np.sum(heatmap):.2f})")
    
    plt.show()

if __name__ == "__main__":
    test_dataloader()