# EdgeCrowdCounter: Real-Time Crowd Counting & Queue Segregation

[cite_start]EdgeCrowdCounter is a custom deep-learning pipeline capable of estimating crowd density in a highly congested indoor environment and mathematically segregating that crowd into distinct physical queues using live CCTV footage[cite: 151].

[cite_start]Designed specifically for real-time edge deployment, this project replaces computationally heavy standard models with a highly optimized MobileNetV3 feature extractor combined with a custom Multifaceted Attention Network (MAN)[cite: 112, 113, 154].

## 🌟 Core Implementation

**Primary Notebook:** `notebook7cd83bb44b (1).ipynb`

This notebook serves as the central engine for the project, containing the end-to-end pipeline:

- [cite_start]**Custom Architecture:** Fuses a MobileNetV3 backbone with Channel, Spatial, and Scale Attention mechanisms to accurately predict crowd density maps[cite: 154].
- [cite_start]**Automated Queue Segregation:** Bypasses expensive camera calibration by implementing high-speed static 2D OpenCV polygon masks to map physical metal railings[cite: 103, 104]. [cite_start]Segregation is calculated dynamically using element-wise multiplication: `Count_i = sum(D * M_i)`[cite: 106].
- [cite_start]**Dynamic Flow Tracking:** Integrates the Lucas-Kanade sparse optical flow algorithm to calculate apparent motion vectors guided by the model's localized density maxima[cite: 233, 235, 236].

## 🧠 Architectural Novelties & Optimizations

### 1. Adaptive Multi-Resolution Feature Fusion

[cite_start]The MobileNetV3 backbone is configured to extract spatial features at three distinct depths (Indices 6, 12, and 16)[cite: 270]. [cite_start]This allows the network to simultaneously process tiny distant heads (1/8 resolution), mid-ground crowds (1/16 resolution), and large foreground individuals (1/32 resolution)[cite: 271].

### 2. k-NN Adaptive Kernels for Perspective Distortion

[cite_start]To address severe scale variations and perspective distortion, the ground-truth annotation pipeline uses a k-Nearest Neighbors (k-NN) approach via SciPy's KDTree[cite: 248, 249]. [cite_start]The Gaussian kernel's standard deviation is dynamically calculated per individual based on the average distance to their 3 nearest neighbors (`sigma = 0.3 * d_bar`)[cite: 250, 251].

### 3. O(W x H) Memory Optimization

[cite_start]Generating density maps for massive 54-megapixel UCF-QNRF images caused severe O(W x H) memory bottlenecks[cite: 266]. [cite_start]This was resolved via a localized bounding box algorithm, reducing operations to O(k_size^2) and achieving a ~10,000x speedup in heatmap generation[cite: 267].

## 🚨 Physics-Based Anomaly Detection

[cite_start]The pipeline bypasses computationally heavy deep-learning anomaly models by leveraging optical flow vectors to calculate real-time crowd kinematics[cite: 238].

- [cite_start]**Panic/Stampede Detection:** Continuously monitors average velocity magnitude[cite: 239]. [cite_start]If average pixel movement exceeds the calibrated threshold (`v > 10.0 pixels/frame`), a high-speed "Panic" anomaly is flagged[cite: 240].
- [cite_start]**Wrong-Way Detection:** Isolates y-axis directional vectors to monitor lane compliance[cite: 241]. [cite_start]Vectors sharply opposing authorized traffic flow (`dy < -5.0`) instantly trigger a "Wrong-Way" security alert[cite: 242].

## 🎥 Results & Demonstrations

Due to GitHub's file size limits, the high-resolution `.mp4` inference outputs are hosted on Google Drive.

- **[Baseline Density Heatmap](https://drive.google.com/file/d/1d4DYvWQH7om-Qyn7rcy9MdUvp-6u0QdK/view?usp=drive_link)** (`output_crowd_heatmap.mp4`): The baseline density estimation map applied over the target environment.
- [cite_start]**[Phase 1: Queue Segregation](https://drive.google.com/file/d/1UGYz19Zq9oEVX8wse5_b193kCp6ryITo/view?usp=drive_link)** (`temple_crowd_output.mp4`): Displays the original video with 9 geometric polygons and a density heatmap overlay, visually demonstrating the localized counting logic[cite: 170].
- [cite_start]**[Phase 2: Dynamic Security](https://drive.google.com/file/d/1KT2siD2FPDp6r1lvd7b9IV0wZErcvIcK/view?usp=drive_link)** (`temple_final_security.mp4`): Showcases autonomous directional flow tracking and physics-based anomaly detection, complete with spatial targeting crosshairs on offending individuals[cite: 246].

## 💾 Pre-Trained Weights

The PyTorch model weights exceed GitHub's 100MB limit and are available for download via Google Drive:

- [cite_start]**[Base Model Weights - 300 Epochs](https://drive.google.com/file/d/1RuItotYrLoFa9vLx5MA7pYS73aKGp5vi/view?usp=drive_link)** (`model_epoch_299.pth`): Trained from scratch on the ShanghaiTech dataset[cite: 156].
- [cite_start]**[Domain-Adapted Edge Weights](https://drive.google.com/file/d/10mFFwFf-tNDckb1TSV6RB8AYVFLC88bZ/view?usp=drive_link)** (`model_temple_finetuned.pth`): Fine-tuned for 50 epochs on a custom dataset to adapt to specific environmental LED lighting and steep camera angles.

## 📂 Repository Structure

- `notebook7cd83bb44b (1).ipynb`: Central execution, architecture, and tracking notebook.
- `annotations.json`: Target domain point annotations used for fine-tuning.
- [cite_start]`/crowdcounting_majorproject`: Directory containing production-ready refactored scripts for massive dataset handling (including `model.py`, `dataset.py`, `train.py`)[cite: 283].
