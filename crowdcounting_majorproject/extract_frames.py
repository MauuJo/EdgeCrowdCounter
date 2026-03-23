import cv2
import os

def extract_frames(video_configs, output_dir="extracted_frames"):
    """
    Extracts frames from multiple videos at specific timestamps.
    """
    # Create the output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)

    for video_path, timestamps in video_configs.items():
        print(f"\n🎬 Processing Video: {video_path}")
        cap = cv2.VideoCapture(video_path)

        if not cap.isOpened():
            print(f"❌ Error: Could not open {video_path}. Check the file path.")
            continue

        fps = cap.get(cv2.CAP_PROP_FPS)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        video_duration = total_frames / fps if fps > 0 else 0
        
        # Extract the video name without the extension for saving
        video_name = os.path.basename(video_path).split('.')[0]

        for timestamp in timestamps:
            if timestamp > video_duration:
                print(f"⚠️ Warning: Timestamp {timestamp}s exceeds video length. Skipping.")
                continue

            # Calculate exact frame index
            target_frame_index = int(timestamp * fps)
            cap.set(cv2.CAP_PROP_POS_FRAMES, target_frame_index)
            success, frame = cap.read()

            if success:
                # Format filename: e.g., "video1_time_012_50s.jpg"
                safe_time_str = f"{timestamp:06.2f}".replace('.', '_')
                output_filepath = os.path.join(output_dir, f"{video_name}_time_{safe_time_str}s.jpg")
                
                cv2.imwrite(output_filepath, frame)
                print(f"✅ Saved: {output_filepath}")
            else:
                print(f"❌ Error: Could not read frame at {timestamp}s")

        cap.release()
    print("\n🎉 All frames extracted successfully!")

# ==========================================
# ENTER YOUR VIDEOS AND TIMESTAMPS HERE
# ==========================================
if __name__ == "__main__":
    # Dictionary format: "Path_to_video.mp4" : [list of timestamps in seconds]
    VIDEO_CONFIGS = {
        "NVR_ch8_main_20260101154514_20260101154638 (1).mp4": [10.5, 45.0, 50.25]   
    }
    
    # Run the extraction
    extract_frames(VIDEO_CONFIGS, output_dir="./my_extracted_frames")