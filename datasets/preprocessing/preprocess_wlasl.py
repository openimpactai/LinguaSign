#!/usr/bin/env python
# datasets/preprocessing/preprocess_wlasl.py
# Script to preprocess the WLASL dataset for training

import os
import json
import cv2
import numpy as np
import argparse
from tqdm import tqdm
import mediapipe as mp
import pickle
from pathlib import Path

# Initialize MediaPipe solutions
mp_holistic = mp.solutions.holistic
mp_drawing = mp.solutions.drawing_utils

def create_directory(directory):
    """Create directory if it doesn't exist."""
    os.makedirs(directory, exist_ok=True)
    print(f"Created directory: {directory}")

def extract_landmarks(video_path, output_path, min_detection_confidence=0.5):
    """
    Extract landmarks from a video using MediaPipe Holistic.
    
    Args:
        video_path: Path to the video file
        output_path: Path to save the extracted landmarks
        min_detection_confidence: Minimum confidence value for detection to be considered successful
    
    Returns:
        True if landmarks were successfully extracted, False otherwise
    """
    # Open the video file
    cap = cv2.VideoCapture(video_path)
    
    # Check if the video was opened successfully
    if not cap.isOpened():
        print(f"Error: Could not open video {video_path}")
        return False
    
    # Initialize MediaPipe Holistic
    with mp_holistic.Holistic(
        min_detection_confidence=min_detection_confidence,
        min_tracking_confidence=0.5) as holistic:
        
        # Initialize lists to store landmarks
        frame_landmarks = []
        
        # Process each frame in the video
        while cap.isOpened():
            success, image = cap.read()
            if not success:
                break
            
            # Convert the BGR image to RGB
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            
            # Process the image and extract landmarks
            results = holistic.process(image)
            
            # Store the landmarks
            frame_data = {
                'pose': None,
                'face': None,
                'left_hand': None,
                'right_hand': None
            }
            
            # Convert landmarks to numpy arrays if they exist
            if results.pose_landmarks:
                pose = np.array([[lm.x, lm.y, lm.z, lm.visibility] for lm in results.pose_landmarks.landmark])
                frame_data['pose'] = pose
            
            if results.face_landmarks:
                face = np.array([[lm.x, lm.y, lm.z] for lm in results.face_landmarks.landmark])
                frame_data['face'] = face
            
            if results.left_hand_landmarks:
                left_hand = np.array([[lm.x, lm.y, lm.z] for lm in results.left_hand_landmarks.landmark])
                frame_data['left_hand'] = left_hand
            
            if results.right_hand_landmarks:
                right_hand = np.array([[lm.x, lm.y, lm.z] for lm in results.right_hand_landmarks.landmark])
                frame_data['right_hand'] = right_hand
            
            frame_landmarks.append(frame_data)
        
        # Release the video capture
        cap.release()
    
    # Save the landmarks
    with open(output_path, 'wb') as f:
        pickle.dump(frame_landmarks, f)
    
    return True

def process_wlasl_dataset(input_dir, output_dir, json_file=None, subset_size=None):
    """
    Process the WLASL dataset to extract landmarks.
    
    Args:
        input_dir: Directory containing the WLASL dataset
        output_dir: Directory to save the processed dataset
        json_file: Path to the WLASL JSON file
        subset_size: Number of classes to include (for testing with smaller subsets)
    """
    # Create the output directory
    create_directory(output_dir)
    create_directory(os.path.join(output_dir, 'landmarks'))
    
    # Find the JSON file if not provided
    if json_file is None:
        for file in os.listdir(input_dir):
            if file.startswith('WLASL_') and file.endswith('.json'):
                json_file = os.path.join(input_dir, file)
                break
        
        if json_file is None:
            print("Error: WLASL JSON file not found in the input directory")
            return
    
    # Load the JSON file
    with open(json_file, 'r') as f:
        data = json.load(f)
    
    # Limit the number of classes if specified
    if subset_size is not None:
        data = data[:subset_size]
    
    # Create a mapping from gloss to index
    gloss_to_index = {item['gloss']: i for i, item in enumerate(data)}
    
    # Save the mapping
    with open(os.path.join(output_dir, 'gloss_to_index.json'), 'w') as f:
        json.dump(gloss_to_index, f)
    
    # Process each sign
    processed_count = 0
    total_videos = sum(len(item['instances']) for item in data)
    
    print(f"Processing {total_videos} videos...")
    
    for item in tqdm(data, desc="Processing classes"):
        gloss = item['gloss']
        
        for instance in item['instances']:
            video_id = instance['video_id']
            
            # Check if the video file exists
            video_path = os.path.join(input_dir, 'videos', f"{video_id}.mp4")
            if not os.path.exists(video_path):
                print(f"Warning: Video {video_id}.mp4 not found")
                continue
            
            # Output path for the landmarks
            landmark_path = os.path.join(output_dir, 'landmarks', f"{video_id}.pkl")
            
            # Extract landmarks from the video
            success = extract_landmarks(video_path, landmark_path)
            if success:
                processed_count += 1
    
    print(f"Processed {processed_count} out of {total_videos} videos")
    
    # Create the metadata file
    metadata = {
        'num_classes': len(data),
        'processed_videos': processed_count,
        'total_videos': total_videos,
        'gloss_to_index': gloss_to_index
    }
    
    with open(os.path.join(output_dir, 'metadata.json'), 'w') as f:
        json.dump(metadata, f)
    
    print("WLASL dataset preprocessing completed!")
    print(f"Processed dataset is available at: {output_dir}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Preprocess WLASL dataset")
    parser.add_argument("--input_dir", type=str, default="../raw/wlasl", 
                        help="Directory containing the WLASL dataset")
    parser.add_argument("--output_dir", type=str, default="../processed/wlasl", 
                        help="Directory to save the processed dataset")
    parser.add_argument("--json_file", type=str, default=None,
                        help="Path to the WLASL JSON file")
    parser.add_argument("--subset_size", type=int, default=None,
                        help="Number of classes to include (for testing with smaller subsets)")
    
    args = parser.parse_args()
    
    # Convert relative paths to absolute paths
    if not os.path.isabs(args.input_dir):
        script_dir = os.path.dirname(os.path.abspath(__file__))
        input_dir = os.path.join(script_dir, args.input_dir)
    else:
        input_dir = args.input_dir
    
    if not os.path.isabs(args.output_dir):
        script_dir = os.path.dirname(os.path.abspath(__file__))
        output_dir = os.path.join(script_dir, args.output_dir)
    else:
        output_dir = args.output_dir
    
    if args.json_file and not os.path.isabs(args.json_file):
        script_dir = os.path.dirname(os.path.abspath(__file__))
        json_file = os.path.join(script_dir, args.json_file)
    else:
        json_file = args.json_file
    
    process_wlasl_dataset(input_dir, output_dir, json_file, args.subset_size)
