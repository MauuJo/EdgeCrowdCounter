import os
import glob
import h5py
import scipy.io as io
import scipy.ndimage
import scipy.spatial
import numpy as np
import cv2
from tqdm import tqdm 


def generate_ucf_adaptive_density_map(img_shape, pts):
    """Generates density map using highly-optimized localized k-NN adaptive math."""
    density_map = np.zeros(img_shape, dtype=np.float32)
    num_points = len(pts)
    
    if num_points == 0:
        return density_map

    # Fast nearest neighbors
    tree = scipy.spatial.KDTree(pts.copy())
    distances, _ = tree.query(pts, k=4)
    
    for i, pt in enumerate(pts):
        pt_x, pt_y = int(pt[0]), int(pt[1])
        
        if pt_y >= img_shape[0] or pt_x >= img_shape[1] or pt_y < 0 or pt_x < 0:
            continue
        
        # 1. Adaptive sigma calculation
        # Fallback to sigma=5 if too few points
        avg_distance = np.mean(distances[i][1:4]) if num_points > 3 else 15.0 
        adaptive_sigma = 0.3 * avg_distance
        adaptive_sigma = np.clip(adaptive_sigma, a_min=1.0, a_max=20.0)
        
        # --- THE SPEED UP: Localized Bounding Box ---
        # 3*sigma safely captures ~99.7% of the Gaussian distribution's spread
        k_size = int(3 * adaptive_sigma) 
        
        # 2. Calculate bounds, ensuring we don't go outside the image borders
        y1 = max(0, pt_y - k_size)
        y2 = min(img_shape[0], pt_y + k_size + 1)
        x1 = max(0, pt_x - k_size)
        x2 = min(img_shape[1], pt_x + k_size + 1)
        
        # 3. Find the local coordinate of the point inside our tiny box
        loc_y = pt_y - y1
        loc_x = pt_x - x1
        
        # 4. Create a tiny localized map, put the point, and blur it
        loc_pt_map = np.zeros((y2 - y1, x2 - x1), dtype=np.float32)
        loc_pt_map[loc_y, loc_x] = 1.0
        
        loc_density = scipy.ndimage.gaussian_filter(loc_pt_map, sigma=adaptive_sigma)
        
        # 5. Add the blurred small patch back into the giant density map
        density_map[y1:y2, x1:x2] += loc_density
        
    return density_map


def process_ucf_qnrf(img_dir, output_dir):
    os.makedirs(output_dir, exist_ok=True)
    
    # CHANGE 1: Removed the [:10] slice at the end of this line
    img_paths = sorted(glob.glob(os.path.join(img_dir, '*.jpg'))) 
    
    print(f"🚀 Starting processing for {len(img_paths)} images...\n")
    
    for img_path in tqdm(img_paths, desc="Generating Heatmaps"):
        img_name = os.path.basename(img_path)
        name, _ = os.path.splitext(img_name)
        
        mat_path = os.path.join(img_dir, f"{name}_ann.mat")
        h5_path = os.path.join(output_dir, f"{name}.h5")
        
        if os.path.exists(h5_path):
            continue 
            
        if not os.path.exists(mat_path):
            continue
            
        img = cv2.imread(img_path)
        if img is None:
            continue
        h, w = img.shape[:2]
        
        # --- THE SAFETY NET ---
        try:
            mat = io.loadmat(mat_path)
            if 'annPoints' in mat:
                pts = mat['annPoints']
            else:
                print(f"⚠️ Skipping {img_name}: No 'annPoints' key found.")
                continue
        except Exception as e:
            print(f"❌ Corrupt .mat file for {img_name}: {e}")
            continue
            
        density_map = generate_ucf_adaptive_density_map((h, w), pts)
        
        with h5py.File(h5_path, 'w') as hf:
            hf.create_dataset('density', data=density_map, compression="gzip")

if __name__ == "__main__":
    # CHANGE 2: Update these paths to point to where the data is on your Workstation!
    # Example: "D:/Datasets/UCF-QNRF_ECCV18/Train"
    WORKSTATION_TRAIN_IMG_DIR = "/path/to/your/workstation/UCF-QNRF_ECCV18/Train" 
    WORKSTATION_OUTPUT_DIR = "/path/to/your/workstation/ucf_train_h5"
    
    process_ucf_qnrf(WORKSTATION_TRAIN_IMG_DIR, WORKSTATION_OUTPUT_DIR)