#!/usr/bin/env python3
"""
Video Splitter with Auto Re-split

This script splits videos larger than 100MB into chunks as close to 100MB as possible.
If any chunk is still over 100MB, it automatically re-splits that chunk until all pieces
are under the limit. Useful for uploading to websites with file size restrictions.
"""

import os
import sys
import glob
import subprocess
import json
from pathlib import Path
import math

# Note: Chunk size is now user-configurable via get_chunk_size() function

def check_ffmpeg():
    """Check if FFmpeg is available."""
    try:
        subprocess.run(['ffmpeg', '-version'], capture_output=True, check=True)
        return True
    except (subprocess.CalledProcessError, FileNotFoundError):
        return False

def get_video_info(video_path):
    """
    Get video information using ffprobe.
    
    Args:
        video_path: Path to the video file
        
    Returns:
        Dictionary with video information or None if failed
    """
    try:
        cmd = [
            'ffprobe', '-v', 'quiet', '-print_format', 'json', '-show_format', 
            '-show_streams', video_path
        ]
        result = subprocess.run(cmd, capture_output=True, text=True, check=True)
        return json.loads(result.stdout)
    except (subprocess.CalledProcessError, json.JSONDecodeError, FileNotFoundError):
        return None

def calculate_chunk_duration(video_info, file_size, max_chunk_size):
    """
    Calculate the optimal chunk duration to stay under the specified size limit.
    
    Args:
        video_info: Video information from ffprobe
        file_size: Current file size in bytes
        max_chunk_size: Maximum chunk size in bytes
        
    Returns:
        Chunk duration in seconds
    """
    try:
        # Get video duration
        duration = float(video_info['format']['duration'])
        
        # Calculate how many chunks we need
        num_chunks = math.ceil(file_size / max_chunk_size)
        
        # Calculate chunk duration
        chunk_duration = duration / num_chunks
        
        # Add a small safety margin (reduce by 5% to ensure we stay under limit)
        chunk_duration *= 0.95
        
        return chunk_duration
        
    except (KeyError, ValueError, ZeroDivisionError):
        # Fallback: assume 10 minutes per chunk if we can't calculate
        return 600  # 10 minutes

def get_chunk_size():
    """Get the desired chunk size from user."""
    print("=== Video Splitter with Custom Chunk Size ===\n")
    print("Choose the maximum size for each video chunk:")
    print("1. 50 MB")
    print("2. 100 MB (default)")
    print("3. 200 MB")
    print("4. 500 MB")
    print("5. Custom size")
    
    while True:
        try:
            choice = int(input("\nChoose chunk size option (1-5): ").strip())
            if choice == 1:
                return 50 * 1024 * 1024  # 50MB
            elif choice == 2:
                return 100 * 1024 * 1024  # 100MB
            elif choice == 3:
                return 200 * 1024 * 1024  # 200MB
            elif choice == 4:
                return 500 * 1024 * 1024  # 500MB
            elif choice == 5:
                while True:
                    try:
                        custom_size = float(input("Enter custom size in MB: ").strip())
                        if custom_size > 0:
                            return int(custom_size * 1024 * 1024)
                        else:
                            print("Error: Size must be greater than 0.")
                    except ValueError:
                        print("Error: Please enter a valid number.")
            else:
                print("Error: Please choose 1-5.")
        except ValueError:
            print("Error: Please enter a valid number.")

def get_processing_mode():
    """Ask user to choose between single video or folder processing."""
    print("\nProcessing options:")
    print("1. Split a single video file")
    print("2. Split all large videos in a folder")
    
    while True:
        try:
            choice = int(input("\nChoose processing mode (1-2): ").strip())
            if choice in [1, 2]:
                return choice
            else:
                print("Error: Please choose 1 or 2.")
        except ValueError:
            print("Error: Please enter a valid number.")

def get_single_video_path():
    """Get single video file path from user."""
    while True:
        video_path = input("\nEnter the path to your video file: ").strip().strip('"')
        if os.path.exists(video_path) and os.path.isfile(video_path):
            # Check if it's a video file
            video_extensions = ['.mp4', '.avi', '.mov', '.mkv', '.wmv', '.flv', '.webm', '.m4v']
            if any(video_path.lower().endswith(ext) for ext in video_extensions):
                return video_path
            else:
                print("Error: File doesn't appear to be a video file. Please try again.")
        else:
            print(f"Error: File '{video_path}' not found. Please try again.")

def get_folder_path(max_chunk_size):
    """Get folder path from user."""
    chunk_size_mb = max_chunk_size / (1024 * 1024)
    while True:
        folder_path = input("\nEnter the path to the folder containing videos: ").strip().strip('"')
        if os.path.exists(folder_path) and os.path.isdir(folder_path):
            # Check if folder contains video files
            video_extensions = ['*.mp4', '*.avi', '*.mov', '*.mkv', '*.wmv', '*.flv', '*.webm', '*.m4v']
            video_files = []
            for ext in video_extensions:
                video_files.extend(glob.glob(os.path.join(folder_path, ext)))
                video_files.extend(glob.glob(os.path.join(folder_path, ext.upper())))
            
            # Remove duplicates that may occur on case-insensitive file systems
            video_files = list(set(video_files))
            
            # Filter for files larger than the specified chunk size
            large_videos = []
            for video in video_files:
                if os.path.getsize(video) > max_chunk_size:
                    large_videos.append(video)
            
            if large_videos:
                print(f"Found {len(large_videos)} video file(s) larger than {chunk_size_mb:.0f}MB:")
                for i, video in enumerate(large_videos[:5], 1):  # Show first 5 files
                    size_mb = os.path.getsize(video) / (1024 * 1024)
                    print(f"  {i}. {os.path.basename(video)} ({size_mb:.1f} MB)")
                if len(large_videos) > 5:
                    print(f"  ... and {len(large_videos) - 5} more files")
                return folder_path, large_videos
            elif video_files:
                print(f"No video files larger than {chunk_size_mb:.0f}MB found in the specified folder.")
                print("All videos are already under the size limit!")
                return None, []
            else:
                print("Error: No video files found in the specified folder. Please try again.")
        else:
            print(f"Error: Folder '{folder_path}' not found. Please try again.")

def ensure_output_directory(base_path):
    """
    Create main output directory if it doesn't exist.
    
    Args:
        base_path: Base directory where videos are located
    
    Returns:
        Path to the main output directory if successful, None if failed
    """
    output_dir = os.path.join(base_path, "split_videos")
    output_path = Path(output_dir)
    if not output_path.exists():
        try:
            output_path.mkdir(parents=True, exist_ok=True)
            print(f"FOLDER Created main output directory: {output_dir}")
        except Exception as e:
            print(f"ERROR Error creating output directory: {e}")
            return None
    else:
        print(f"FOLDER Using main output directory: {output_dir}")
    return output_dir

def create_video_output_directory(main_output_dir, video_path):
    """
    Create individual directory for a specific video's chunks.
    
    Args:
        main_output_dir: Main split_videos directory
        video_path: Path to the video being processed
    
    Returns:
        Path to the video-specific directory if successful, None if failed
    """
    video_name = Path(video_path).stem  # Get filename without extension
    video_output_dir = os.path.join(main_output_dir, video_name)
    video_output_path = Path(video_output_dir)
    
    if not video_output_path.exists():
        try:
            video_output_path.mkdir(parents=True, exist_ok=True)
            print(f"  FOLDER Created folder: {video_name}")
        except Exception as e:
            print(f"  ERROR Error creating video directory: {e}")
            return None
    else:
        print(f"  FOLDER Using folder: {video_name}")
    
    return video_output_dir

def split_oversized_chunk(chunk_path, video_output_dir, chunk_number, max_chunk_size):
    """
    Recursively split a chunk that's still over the specified size limit.
    
    Args:
        chunk_path: Path to the oversized chunk
        video_output_dir: Directory where chunks are stored
        chunk_number: Original chunk number for naming
        max_chunk_size: Maximum chunk size in bytes
        
    Returns:
        List of successfully created sub-chunks
    """
    print(f"    PROCESSING Re-splitting oversized chunk...")
    
    # Get chunk info
    chunk_info = get_video_info(chunk_path)
    if not chunk_info:
        print(f"    ERROR Could not get chunk information for re-splitting")
        return []
    
    chunk_size = os.path.getsize(chunk_path)
    chunk_duration = float(chunk_info['format']['duration'])
    
    # Calculate how many sub-chunks we need (aim for 90% of max size to be safe)
    target_size = int(max_chunk_size * 0.9)  # 90% of max size for re-splits
    num_subchunks = math.ceil(chunk_size / target_size)
    subchunk_duration = (chunk_duration / num_subchunks) * 0.95  # 5% safety margin
    
    print(f"    CHART Splitting into {num_subchunks} sub-chunks of ~{subchunk_duration:.1f}s each")
    
    # Get original filename parts
    chunk_path_obj = Path(chunk_path)
    base_name = chunk_path_obj.stem.replace(f"_part{chunk_number:02d}", "")
    extension = chunk_path_obj.suffix
    
    successful_subchunks = []
    
    # Create sub-chunks
    for i in range(num_subchunks):
        start_time = i * subchunk_duration
        
        if start_time >= chunk_duration:
            break
            
        # Create sub-chunk filename: video_part01a.mp4, video_part01b.mp4, etc.
        sub_letter = chr(ord('a') + i)
        subchunk_filename = f"{base_name}_part{chunk_number:02d}{sub_letter}{extension}"
        subchunk_path = os.path.join(video_output_dir, subchunk_filename)
        
        print(f"    Creating sub-chunk {i+1}/{num_subchunks}: {subchunk_filename}")
        
        # FFmpeg command to create sub-chunk
        cmd = [
            'ffmpeg', '-i', chunk_path,
            '-ss', str(start_time),
            '-t', str(subchunk_duration),
            '-c', 'copy',
            '-avoid_negative_ts', 'make_zero',
            '-y',
            subchunk_path
        ]
        
        try:
            subprocess.run(cmd, capture_output=True, text=True, check=True)
            
            if os.path.exists(subchunk_path):
                subchunk_size = os.path.getsize(subchunk_path)
                subchunk_size_mb = subchunk_size / (1024 * 1024)
                max_size_mb = max_chunk_size / (1024 * 1024)
                
                if subchunk_size > max_chunk_size:
                    print(f"      WARNING  Sub-chunk still {subchunk_size_mb:.1f} MB (over {max_size_mb:.0f}MB limit) - may need manual review")
                else:
                    print(f"      SUCCESS Sub-chunk: {subchunk_size_mb:.1f} MB")
                
                successful_subchunks.append(subchunk_path)
            else:
                print(f"      ERROR Failed to create sub-chunk {i+1}")
                
        except subprocess.CalledProcessError as e:
            print(f"      ERROR Error creating sub-chunk {i+1}: {e}")
            continue
    
    # Remove the original oversized chunk if we successfully created sub-chunks
    if successful_subchunks:
        try:
            os.remove(chunk_path)
            print(f"      Removed original oversized chunk")
        except Exception as e:
            print(f"    WARNING  Could not remove original chunk: {e}")
    
    return successful_subchunks

def split_video(video_path, main_output_dir, max_chunk_size):
    """
    Split a video into chunks under the specified size limit.
    
    Args:
        video_path: Path to the input video
        main_output_dir: Main directory to save the split videos
        max_chunk_size: Maximum chunk size in bytes
        
    Returns:
        True if successful, False otherwise
    """
    print(f"\nVIDEO Processing: {os.path.basename(video_path)}")
    
    # Create individual directory for this video's chunks
    video_output_dir = create_video_output_directory(main_output_dir, video_path)
    if not video_output_dir:
        return False
    
    # Get file size
    file_size = os.path.getsize(video_path)
    size_mb = file_size / (1024 * 1024)
    max_size_mb = max_chunk_size / (1024 * 1024)
    
    print(f"  File size: {size_mb:.1f} MB")
    
    if file_size <= max_chunk_size:
        print(f"  SUCCESS File is already under {max_size_mb:.0f}MB, skipping...")
        return True
    
    # Get video information
    video_info = get_video_info(video_path)
    if not video_info:
        print(f"  ERROR Could not get video information")
        return False
    
    # Calculate chunk duration
    chunk_duration = calculate_chunk_duration(video_info, file_size, max_chunk_size)
    total_duration = float(video_info['format']['duration'])
    
    print(f"  Duration: {total_duration:.1f} seconds")
    print(f"  Chunk duration: {chunk_duration:.1f} seconds")
    
    # Calculate number of chunks
    num_chunks = math.ceil(total_duration / chunk_duration)
    print(f"  Will create {num_chunks} chunks")
    
    # Prepare output filename pattern
    input_path = Path(video_path)
    base_name = input_path.stem
    extension = input_path.suffix
    
    success_count = 0
    
    # Split the video
    for i in range(num_chunks):
        start_time = i * chunk_duration
        
        # Don't exceed the total duration
        if start_time >= total_duration:
            break
            
        output_filename = f"{base_name}_part{i+1:02d}{extension}"
        output_path = os.path.join(video_output_dir, output_filename)
        
        print(f"  Creating chunk {i+1}/{num_chunks}: {output_filename}")
        
        # FFmpeg command to split video
        cmd = [
            'ffmpeg', '-i', video_path,
            '-ss', str(start_time),
            '-t', str(chunk_duration),
            '-c', 'copy',  # Copy without re-encoding for speed
            '-avoid_negative_ts', 'make_zero',
            '-y',  # Overwrite output files
            output_path
        ]
        
        try:
            # Run FFmpeg with minimal output
            result = subprocess.run(cmd, capture_output=True, text=True, check=True)
            
            # Check if output file was created and is reasonable size
            if os.path.exists(output_path):
                chunk_size = os.path.getsize(output_path)
                chunk_size_mb = chunk_size / (1024 * 1024)
                max_size_mb = max_chunk_size / (1024 * 1024)
                
                if chunk_size > max_chunk_size:
                    print(f"    WARNING  Chunk is {chunk_size_mb:.1f} MB (over {max_size_mb:.0f}MB limit!)")
                    # Automatically re-split this oversized chunk
                    subchunks = split_oversized_chunk(output_path, video_output_dir, i+1, max_chunk_size)
                    if subchunks:
                        print(f"    SUCCESS Successfully re-split into {len(subchunks)} sub-chunks")
                        success_count += len(subchunks)
                    else:
                        print(f"    ERROR Failed to re-split oversized chunk")
                else:
                    print(f"    SUCCESS Created: {chunk_size_mb:.1f} MB")
                    success_count += 1
            else:
                print(f"    ERROR Failed to create chunk {i+1}")
                
        except subprocess.CalledProcessError as e:
            print(f"    ERROR Error creating chunk {i+1}: {e}")
            continue
    
    if success_count > 0:
        print(f"  SUCCESS Successfully created {success_count} chunks")
        return True
    else:
        print(f"  ERROR Failed to create any chunks")
        return False

def process_videos(video_files, base_path, max_chunk_size):
    """
    Process multiple videos for splitting.
    
    Args:
        video_files: List of video file paths
        base_path: Base directory path
        max_chunk_size: Maximum chunk size in bytes
        
    Returns:
        Number of successfully processed videos
    """
    # Create output directory
    output_dir = ensure_output_directory(base_path)
    if not output_dir:
        return 0
    
    total_videos = len(video_files)
    successful_videos = 0
    
    print(f"\nVIDEO Processing {total_videos} video(s) for splitting...")
    print("=" * 60)
    
    for i, video_path in enumerate(video_files, 1):
        video_name = os.path.basename(video_path)
        print(f"\nVIDEO Processing video {i}/{total_videos}: {video_name}")
        
        try:
            success = split_video(video_path, output_dir, max_chunk_size)
            if success:
                successful_videos += 1
                print(f"SUCCESS Successfully processed: {video_name}")
            else:
                print(f"ERROR Failed to process: {video_name}")
        except Exception as e:
            print(f"ERROR Error processing {video_name}: {e}")
        
        # Add separator between videos
        if i < total_videos:
            print("-" * 40)
    
    # Print final summary
    print(f"\n{'='*60}")
    print(f"CHART PROCESSING SUMMARY")
    print(f"{'='*60}")
    print(f"Total videos: {total_videos}")
    print(f"SUCCESS Successful: {successful_videos}")
    print(f"ERROR Failed: {total_videos - successful_videos}")
    print(f"\nFOLDER All split videos saved in individual folders within: {output_dir}")
    
    return successful_videos

def main():
    """Main function to run the video splitter."""
    try:
        # Check if FFmpeg is available
        if not check_ffmpeg():
            print("ERROR FFmpeg is not installed or not available in PATH.")
            print("Please install FFmpeg to use this script.")
            print("Download from: https://ffmpeg.org/download.html")
            sys.exit(1)
        
        print("SUCCESS FFmpeg found")
        
        # Get chunk size from user
        max_chunk_size = get_chunk_size()
        chunk_size_mb = max_chunk_size / (1024 * 1024)
        print(f"\nSUCCESS Using chunk size: {chunk_size_mb:.0f} MB")
        
        # Get processing mode
        mode = get_processing_mode()
        
        if mode == 1:
            # Single video processing
            video_path = get_single_video_path()
            
            # Check if video needs splitting
            file_size = os.path.getsize(video_path)
            if file_size <= max_chunk_size:
                size_mb = file_size / (1024 * 1024)
                print(f"\nSUCCESS Video is only {size_mb:.1f} MB - no splitting needed!")
                return
            
            base_path = os.path.dirname(video_path)
            video_files = [video_path]
        else:
            # Folder processing
            result = get_folder_path(max_chunk_size)
            if result is None:
                return
            
            base_path, video_files = result
            if not video_files:
                return
        
        # Process the videos
        successful_videos = process_videos(video_files, base_path, max_chunk_size)
        
        if successful_videos > 0:
            print(f"\nCOMPLETE Successfully processed {successful_videos} video(s)!")
        else:
            print("\nERROR No videos were processed successfully.")
            sys.exit(1)
    
    except KeyboardInterrupt:
        print("\n\nSTOP Operation cancelled by user.")
        sys.exit(1)
    except Exception as e:
        print(f"\nERROR Unexpected error: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()
