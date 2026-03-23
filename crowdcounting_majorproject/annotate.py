import cv2
import os
import glob
import numpy as np
import scipy.ndimage
import scipy.spatial 
import h5py

# Global variables
points = []
clone = None

def click_and_mark(event, x, y, flags, param):
    """Mouse callback function to capture clicks."""
    global points, clone
    if event == cv2.EVENT_LBUTTONDOWN:
        points.append((x, y))
        cv2.circle(clone, (x, y), 3, (0, 0, 255), -1)
        cv2.imshow("Annotator", clone)

def generate_adaptive_density_map(img_shape, pts):
    """Generates a density map using k-NN adaptive math for scale variation."""
    density_map = np.zeros(img_shape, dtype=np.float32)
    num_points = len(pts)
    
    if num_points == 0:
        return density_map

    if num_points <= 3:
        for p in pts:
            density_map[int(p[1]), int(p[0])] = 1.0
        return scipy.ndimage.gaussian_filter(density_map, sigma=5)

    tree = scipy.spatial.KDTree(pts.copy())
    distances, _ = tree.query(pts, k=4)
    
    for i, pt in enumerate(pts):
        pt_x, pt_y = int(pt[0]), int(pt[1])
        
        if pt_y >= img_shape[0] or pt_x >= img_shape[1] or pt_y < 0 or pt_x < 0:
            continue
            
        pt_map = np.zeros(img_shape, dtype=np.float32)
        pt_map[pt_y, pt_x] = 1.0
        
        # Adaptive sigma calculation based on nearest neighbors
        avg_distance = np.mean(distances[i][1:4])
        adaptive_sigma = 0.3 * avg_distance
        # a_min set to 1.0 to handle very small heads in the background
        adaptive_sigma = np.clip(adaptive_sigma, a_min=1.0, a_max=20.0)
        
        density_map += scipy.ndimage.gaussian_filter(pt_map, sigma=adaptive_sigma)
        
    return density_map

def annotate_and_generate_heatmaps(img_dir, output_gt_dir):
    """Loops through images, allows annotation, and generates .h5 heatmaps."""
    global points, clone
    os.makedirs(output_gt_dir, exist_ok=True)
    
    image_paths = sorted(glob.glob(os.path.join(img_dir, '*.jpg')))
    
    if not image_paths:
        print(f"❌ No .jpg images found in {img_dir}. Did you run the extraction script?")
        return
        
    print(f"✅ Found {len(image_paths)} images to annotate.")
    print("\n--- CONTROLS ---")
    print("🖱️ LEFT CLICK   : Mark a head")
    print("⌨️ 'u'          : Undo the last click")
    print("⌨️ SPACE or 'n' : Save the heatmap and go to the Next image")
    print("⌨️ ESC or 'q'   : Save current and Quit entirely")
    print("----------------\n")

    for img_path in image_paths:
        filename = os.path.basename(img_path)
        name, _ = os.path.splitext(filename)
        h5_path = os.path.join(output_gt_dir, f"{name}.h5")
        
        if os.path.exists(h5_path):
            print(f"⏭️ Skipping {filename} (Already annotated)")
            continue
            
        original_img = cv2.imread(img_path)
        if original_img is None:
            continue
            
        # Resize to match inference size
        original_img = cv2.resize(original_img, (1280, 720))
        clone = original_img.copy()
        points = [] 
        
        cv2.namedWindow("Annotator", cv2.WINDOW_NORMAL)
        cv2.setMouseCallback("Annotator", click_and_mark)
        
        while True:
            cv2.imshow("Annotator", clone)
            key = cv2.waitKey(1) & 0xFF
            
            if key == ord("u") and len(points) > 0:
                points.pop()
                clone = original_img.copy() 
                for p in points:
                    cv2.circle(clone, p, 3, (0, 0, 255), -1)
                print("Undo last point.")
                
            elif key == ord("n") or key == ord(" ") or key == 32:
                break
                
            elif key == ord("q") or key == 27: 
                print("\n🛑 Exiting annotation tool...")
                cv2.destroyAllWindows()
                return 

        # Generate and save
        h, w = original_img.shape[:2]
        pts_array = np.array(points)
        final_density_map = generate_adaptive_density_map((h, w), pts_array)
        
        with h5py.File(h5_path, 'w') as hf:
            hf['density'] = final_density_map
            
        print(f"💾 Saved {h5_path} with {len(points)} annotations.")

    cv2.destroyAllWindows()
    print("\n🎉 All frames annotated successfully!")

if __name__ == "__main__":
    INPUT_FRAMES_DIR = "./my_extracted_frames" 
    OUTPUT_HEATMAPS_DIR = "./my_extracted_ground_truth" 
    annotate_and_generate_heatmaps(INPUT_FRAMES_DIR, OUTPUT_HEATMAPS_DIR)