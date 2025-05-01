#!/usr/bin/env python3
import os
import argparse
import subprocess
import math
import sys
from pathlib import Path

def get_video_size(video_path):
    """Return the size of the video in MB."""
    try:
        return os.path.getsize(video_path) / (1024 * 1024)
    except FileNotFoundError:
        print(f"Error: File {video_path} not found")
        return 0

def ensure_valid_extension(path):
    """Ensure the file has a valid video extension."""
    valid_extensions = ['.mp4', '.avi', '.mkv', '.mov', '.webm']
    path_obj = Path(path)
    
    if path_obj.suffix.lower() not in valid_extensions:
        return str(path_obj) + '.mp4'
    return path

def compress_video(input_path, output_path, target_size_mb=10, min_bitrate=500, max_bitrate=8000):
    """Compress video to target size in MB."""
    output_path = ensure_valid_extension(output_path)
    
    if not os.path.exists(input_path):
        print(f"Error: Input file {input_path} not found")
        return
    
    input_size = get_video_size(input_path)
    print(f"Original size: {input_size:.2f} MB")
    
    if input_size <= target_size_mb:
        print(f"Video is already smaller than {target_size_mb} MB, no compression needed")
        if input_path != output_path:
            subprocess.run(['cp', input_path, output_path])
        return
    
    try:
        cmd = ['ffprobe', '-v', 'error', '-show_entries', 'format=duration', 
               '-of', 'default=noprint_wrappers=1:nokey=1', input_path]
        duration = float(subprocess.check_output(cmd).decode('utf-8').strip())
        
        target_bits = target_size_mb * 8192  # MB to kilobits
        bitrate = int(target_bits / duration * 0.93)
        bitrate = max(min_bitrate, min(bitrate, max_bitrate))
        
        print(f"Target bitrate: {bitrate} kbps")
        
        output_dir = os.path.dirname(output_path)
        if output_dir and not os.path.exists(output_dir):
            os.makedirs(output_dir)
        
        print("Running first pass...")
        cmd1 = [
            'ffmpeg', '-y', '-i', input_path, '-c:v', 'libx264', '-b:v', f'{bitrate}k',
            '-pass', '1', '-f', 'mp4', '-an', '-v', 'quiet', '/dev/null'
        ]
        
        subprocess.run(cmd1, check=True)
        
        print("Running second pass...")
        cmd2 = [
            'ffmpeg', '-y', '-i', input_path, '-c:v', 'libx264', '-b:v', f'{bitrate}k',
            '-pass', '2', '-c:a', 'aac', '-b:a', '128k', output_path
        ]
        
        subprocess.run(cmd2, check=True)
        
        if os.path.exists(output_path):
            output_size = get_video_size(output_path)
            print(f"Final size: {output_size:.2f} MB")
        else:
            print(f"Warning: Output file {output_path} was not created")
        
        for temp_file in Path('.').glob('ffmpeg2pass*'):
            temp_file.unlink()
    
    except subprocess.CalledProcessError as e:
        print(f"FFmpeg error: {e}")
    except Exception as e:
        print(f"Unexpected error: {e}")

def main():
    parser = argparse.ArgumentParser(description='Compress video to target size')
    parser.add_argument('--input', '-i', required=True, help='Input video path')
    parser.add_argument('--output', '-o', required=True, help='Output video path')
    parser.add_argument('--target-size', '-t', type=float, default=10.0, 
                        help='Target size in MB (default: 10)')
    
    args = parser.parse_args()
    
    compress_video(args.input, args.output, args.target_size)

if __name__ == '__main__':
    main() 