import os
import glob
import cv2
import h5py
import torch
import numpy as np
import random
from torch.utils.data import Dataset
from torchvision import transforms

class CrowdDataset(Dataset):
    def __init__(self, img_dir, h5_dir, crop_size=512, downsample_ratio=8):
        self.crop_size = crop_size
        self.downsample_ratio = downsample_ratio
        
        # Safely grab ONLY images that have a matching .h5 file
        all_imgs = sorted(glob.glob(os.path.join(img_dir, '*.jpg')))
        self.img_paths = [
            img for img in all_imgs 
            if os.path.exists(os.path.join(h5_dir, os.path.basename(img).replace('.jpg', '.h5')))
        ]
        
        self.h5_dir = h5_dir
        self.transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

    def __len__(self):
        return len(self.img_paths)

    def __getitem__(self, idx):
        img_path = self.img_paths[idx]
        img = cv2.imread(img_path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        
        h5_path = os.path.join(self.h5_dir, os.path.basename(img_path).replace('.jpg', '.h5'))
        with h5py.File(h5_path, 'r') as hf:
            density_map = np.array(hf['density'])

        h, w = img.shape[:2]

        if h < self.crop_size or w < self.crop_size:
            pad_h = max(0, self.crop_size - h)
            pad_w = max(0, self.crop_size - w)
            img = np.pad(img, ((0, pad_h), (0, pad_w), (0, 0)), mode='constant')
            density_map = np.pad(density_map, ((0, pad_h), (0, pad_w)), mode='constant')
            h, w = img.shape[:2]

        x1 = random.randint(0, w - self.crop_size)
        y1 = random.randint(0, h - self.crop_size)
        img_crop = img[y1:y1+self.crop_size, x1:x1+self.crop_size]
        density_crop = density_map[y1:y1+self.crop_size, x1:x1+self.crop_size]

        target_h = self.crop_size // self.downsample_ratio
        target_w = self.crop_size // self.downsample_ratio
        original_count = np.sum(density_crop) 
        
        density_crop_resized = cv2.resize(density_crop, (target_w, target_h), interpolation=cv2.INTER_CUBIC)
        resized_count = np.sum(density_crop_resized)
        if resized_count > 0:
            density_crop_resized = density_crop_resized * (original_count / resized_count)

        img_tensor = self.transform(img_crop)
        density_tensor = torch.from_numpy(density_crop_resized).unsqueeze(0).float()

        return img_tensor, density_tensor