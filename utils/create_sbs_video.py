import cv2
import numpy as np
import argparse

def create_combined_video(video_path1, video_path2, text1, text2, output_path="combined_video.mp4"):
    cap1 = cv2.VideoCapture(video_path1)
    cap2 = cv2.VideoCapture(video_path2)
    
    if not cap1.isOpened() or not cap2.isOpened():
        print("Error opening video files")
        return
    
    width1 = int(cap1.get(cv2.CAP_PROP_FRAME_WIDTH))
    height1 = int(cap1.get(cv2.CAP_PROP_FRAME_HEIGHT))
    width2 = int(cap2.get(cv2.CAP_PROP_FRAME_WIDTH))
    height2 = int(cap2.get(cv2.CAP_PROP_FRAME_HEIGHT))
    
    target_height = min(height1, height2)
    
    new_width1 = int(width1 * (target_height / height1))
    new_width2 = int(width2 * (target_height / height2))
    
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 5
    font_color = (255, 255, 255)
    font_thickness = 5
    text_height = 200
    
    fps = min(cap1.get(cv2.CAP_PROP_FPS), cap2.get(cv2.CAP_PROP_FPS))
    combined_width = new_width1 + new_width2
    combined_height = target_height + text_height
    
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, fps, (combined_width, combined_height))
    
    while True:
        ret1, frame1 = cap1.read()
        ret2, frame2 = cap2.read()
        
        if not ret1 or not ret2:
            break
        
        frame1_resized = cv2.resize(frame1, (new_width1, target_height))
        frame2_resized = cv2.resize(frame2, (new_width2, target_height))
        
        combined_frame = np.zeros((combined_height, combined_width, 3), dtype=np.uint8)
        
        combined_frame[:target_height, :new_width1] = frame1_resized
        combined_frame[:target_height, new_width1:] = frame2_resized
        
        text1_size = cv2.getTextSize(text1, font, font_scale, font_thickness)[0]
        text1_x = int(new_width1/2 - text1_size[0]/2)
        text1_y = target_height + 35
        
        text2_size = cv2.getTextSize(text2, font, font_scale, font_thickness)[0]
        text2_x = int(new_width1 + new_width2/2 - text2_size[0]/2)
        text2_y = target_height + 35
        
        cv2.putText(combined_frame, text1, (text1_x, text1_y), font, font_scale, font_color, font_thickness)
        cv2.putText(combined_frame, text2, (text2_x, text2_y), font, font_scale, font_color, font_thickness)
        
        out.write(combined_frame)
    
    cap1.release()
    cap2.release()
    out.release()
    print(f"Video successfully created: {output_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Create SBS video from SAM2 and exp results')
    parser.add_argument('--sam2-result', help='Path to SAM2 result')
    parser.add_argument('--exp-result', help='Path to exp result')
    parser.add_argument('--exp-name', help='Experiment name')
    parser.add_argument('--output_path', default='combined_video.mp4', help='Path to save result')
    
    args = parser.parse_args()
    
    create_combined_video(
        args.sam2_result,
        args.exp_result,
        "SAM2",
        args.exp_name,
        args.output_path
    )