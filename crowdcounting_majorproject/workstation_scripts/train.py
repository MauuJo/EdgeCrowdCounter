import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from model import EdgeCrowdCounter
from dataset import CrowdDataset

# ⚠️ UPDATE THESE PATHS TO YOUR WORKSTATION FOLDERS
TRAIN_IMG_DIR = "D:/Path/To/UCF-QNRF/Train" 
TRAIN_H5_DIR  = "D:/Path/To/UCF-QNRF/Train_h5"

def train():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"🖥️ Training on: {device}")
    
    dataset = CrowdDataset(img_dir=TRAIN_IMG_DIR, h5_dir=TRAIN_H5_DIR, crop_size=512)
    
    # batch_size=8 is good for 8GB+ GPUs. num_workers=4 speeds up CPU data loading.
    loader = DataLoader(dataset, batch_size=8, shuffle=True, num_workers=4, pin_memory=True)
    
    model = EdgeCrowdCounter().to(device)
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=1e-4)
    
    epochs = 150
    best_loss = float('inf')
    
    print(f"🚀 Starting 10-Hour Training Run for {epochs} Epochs...")
    
    for epoch in range(epochs):
        model.train()
        epoch_loss = 0.0
        
        for batch_idx, (images, heatmaps) in enumerate(loader):
            images, heatmaps = images.to(device), heatmaps.to(device)
            
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, heatmaps)
            loss.backward()
            optimizer.step()
            
            epoch_loss += loss.item()
            
        avg_loss = epoch_loss / len(loader)
        print(f"📈 Epoch [{epoch+1}/{epochs}] | Avg MSE Loss: {avg_loss:.6f}")
        
        # --- CRUCIAL: SAVE BEST MODEL ---
        if avg_loss < best_loss:
            best_loss = avg_loss
            torch.save(model.state_dict(), 'best_edge_crowd_model.pth')
            print(f"💾 New Best Model Saved! (Loss: {best_loss:.6f})")

if __name__ == "__main__":
    train()