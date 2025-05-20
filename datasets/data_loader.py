#!/usr/bin/env python
# datasets/data_loader.py
# Data loader for sign language datasets

import os
import json
import pickle
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from pathlib import Path
import random
import cv2

class SignLanguageDataset(Dataset):
    """
    Dataset class for sign language data, compatible with PyTorch DataLoader.
    Loads preprocessed landmark features extracted with MediaPipe.
    """
    def __init__(self, 
                 data_dir, 
                 split='train', 
                 transform=None, 
                 max_seq_length=100,
                 min_seq_length=10,
                 random_seed=42):
        """
        Initialize the dataset.
        
        Args:
            data_dir: Directory containing the processed dataset
            split: Data split to use ('train', 'val', or 'test')
            transform: Optional transforms to apply to the data
            max_seq_length: Maximum sequence length to use (longer sequences will be truncated)
            min_seq_length: Minimum sequence length required (shorter sequences will be padded)
            random_seed: Random seed for reproducibility
        """
        self.data_dir = data_dir
        self.split = split
        self.transform = transform
        self.max_seq_length = max_seq_length
        self.min_seq_length = min_seq_length
        
        # Set random seed for reproducibility
        random.seed(random_seed)
        np.random.seed(random_seed)
        
        # Load metadata
        metadata_path = os.path.join(data_dir, 'metadata.json')
        if not os.path.exists(metadata_path):
            raise FileNotFoundError(f"Metadata file not found at {metadata_path}")
        
        with open(metadata_path, 'r') as f:
            self.metadata = json.load(f)
        
        # Load gloss to index mapping
        self.gloss_to_index = self.metadata['gloss_to_index']
        self.num_classes = len(self.gloss_to_index)
        
        # Load split information
        splits_dir = os.path.join(data_dir, 'splits')
        if not os.path.exists(splits_dir):
            # If splits directory doesn't exist, create splits
            self._create_splits()
        
        split_file = os.path.join(splits_dir, f"{split}.json")
        if not os.path.exists(split_file):
            raise FileNotFoundError(f"Split file not found at {split_file}")
        
        with open(split_file, 'r') as f:
            self.samples = json.load(f)
        
        # Load the actual data
        self.landmarks_dir = os.path.join(data_dir, 'landmarks')
        if not os.path.exists(self.landmarks_dir):
            raise FileNotFoundError(f"Landmarks directory not found at {self.landmarks_dir}")
    
    def _create_splits(self, train_ratio=0.8, val_ratio=0.1, test_ratio=0.1):
        """
        Create train/val/test splits if they don't exist.
        
        Args:
            train_ratio: Proportion of data to use for training
            val_ratio: Proportion of data to use for validation
            test_ratio: Proportion of data to use for testing
        """
        # Ensure the ratios sum to 1
        assert abs(train_ratio + val_ratio + test_ratio - 1.0) < 1e-5, "Split ratios must sum to 1"
        
        # Create the splits directory
        splits_dir = os.path.join(self.data_dir, 'splits')
        os.makedirs(splits_dir, exist_ok=True)
        
        # Get all landmark files
        landmark_files = [f for f in os.listdir(self.landmarks_dir) if f.endswith('.pkl')]
        random.shuffle(landmark_files)
        
        # Determine split indices
        n_samples = len(landmark_files)
        n_train = int(n_samples * train_ratio)
        n_val = int(n_samples * val_ratio)
        
        # Create the splits
        train_files = landmark_files[:n_train]
        val_files = landmark_files[n_train:n_train+n_val]
        test_files = landmark_files[n_train+n_val:]
        
        # Save the splits
        splits = {
            'train': train_files,
            'val': val_files,
            'test': test_files
        }
        
        for split_name, files in splits.items():
            split_path = os.path.join(splits_dir, f"{split_name}.json")
            with open(split_path, 'w') as f:
                json.dump(files, f)
        
        print(f"Created splits: {n_train} train, {n_val} val, {len(test_files)} test")
    
    def __len__(self):
        """Return the number of samples in the dataset."""
        return len(self.samples)
    
    def __getitem__(self, idx):
        """
        Get a sample from the dataset.
        
        Args:
            idx: Index of the sample to get
        
        Returns:
            Dictionary containing the sample data
        """
        # Get the file name
        file_name = self.samples[idx]
        file_path = os.path.join(self.landmarks_dir, file_name)
        
        # Load the landmarks
        with open(file_path, 'rb') as f:
            landmarks = pickle.load(f)
        
        # Extract the video ID and gloss from the file name
        video_id = os.path.splitext(file_name)[0]
        
        # Find the gloss for this video
        gloss = None
        for g, idx in self.gloss_to_index.items():
            # Check if this gloss has this video
            for item in self.metadata['instances']:
                if item['video_id'] == video_id and item['gloss'] == g:
                    gloss = g
                    break
            if gloss:
                break
        
        if gloss is None:
            # If gloss is not found, use a dummy label
            label = 0
        else:
            # Convert gloss to label index
            label = self.gloss_to_index[gloss]
        
        # Process the landmarks
        processed_data = self._process_landmarks(landmarks)
        
        # Apply transform if specified
        if self.transform:
            processed_data = self.transform(processed_data)
        
        return {
            'features': processed_data,
            'label': label,
            'video_id': video_id
        }
    
    def _process_landmarks(self, landmarks):
        """
        Process the landmarks for model input.
        
        Args:
            landmarks: List of dictionaries containing landmarks for each frame
        
        Returns:
            Processed landmarks as a numpy array
        """
        # Limit sequence length
        seq_length = min(len(landmarks), self.max_seq_length)
        landmarks = landmarks[:seq_length]
        
        # Initialize arrays to store processed landmarks
        processed_landmarks = []
        
        for frame in landmarks:
            # Extract relevant features (hands, pose, face)
            frame_features = []
            
            # Add hand landmarks if they exist
            if frame['left_hand'] is not None:
                frame_features.append(frame['left_hand'].flatten())
            else:
                # Use zeros if hand is not detected
                frame_features.append(np.zeros(21 * 3))  # 21 landmarks, 3 coordinates per landmark
            
            if frame['right_hand'] is not None:
                frame_features.append(frame['right_hand'].flatten())
            else:
                frame_features.append(np.zeros(21 * 3))
            
            # Add pose landmarks if they exist (only upper body)
            if frame['pose'] is not None:
                # Extract only upper body landmarks (first 25 landmarks)
                upper_body = frame['pose'][:25, :3]  # Exclude visibility
                frame_features.append(upper_body.flatten())
            else:
                frame_features.append(np.zeros(25 * 3))
            
            # Concatenate all features
            processed_landmarks.append(np.concatenate(frame_features))
        
        # Pad sequence if it's shorter than min_seq_length
        if seq_length < self.min_seq_length:
            padding = [np.zeros_like(processed_landmarks[0]) for _ in range(self.min_seq_length - seq_length)]
            processed_landmarks.extend(padding)
            seq_length = self.min_seq_length
        
        # Convert to numpy array
        processed_landmarks = np.array(processed_landmarks)
        
        return processed_landmarks

def get_dataloader(data_dir, batch_size=32, split='train', num_workers=4, **kwargs):
    """
    Get a DataLoader for the specified dataset.
    
    Args:
        data_dir: Directory containing the processed dataset
        batch_size: Batch size
        split: Data split to use ('train', 'val', or 'test')
        num_workers: Number of worker processes for data loading
        **kwargs: Additional arguments to pass to SignLanguageDataset
    
    Returns:
        PyTorch DataLoader
    """
    dataset = SignLanguageDataset(data_dir, split=split, **kwargs)
    
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=(split == 'train'),
        num_workers=num_workers,
        pin_memory=True
    )
    
    return dataloader

if __name__ == "__main__":
    # Test the data loader
    import argparse
    
    parser = argparse.ArgumentParser(description="Test the sign language data loader")
    parser.add_argument("--data_dir", type=str, required=True, 
                        help="Directory containing the processed dataset")
    parser.add_argument("--batch_size", type=int, default=32, 
                        help="Batch size")
    
    args = parser.parse_args()
    
    # Get the data loader
    dataloader = get_dataloader(args.data_dir, batch_size=args.batch_size)
    
    # Print information about the dataset
    print(f"Dataset size: {len(dataloader.dataset)}")
    print(f"Number of batches: {len(dataloader)}")
    
    # Get a sample batch
    for batch in dataloader:
        features = batch['features']
        labels = batch['label']
        video_ids = batch['video_id']
        
        print(f"Features shape: {features.shape}")
        print(f"Labels shape: {labels.shape}")
        print(f"Sample video IDs: {video_ids[:5]}")
        break
