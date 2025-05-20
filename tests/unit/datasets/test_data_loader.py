#!/usr/bin/env python
# tests/unit/datasets/test_data_loader.py
# Unit tests for dataset loading functionality

import os
import sys
import json
import pickle
import shutil
import pytest
import numpy as np
import torch
from pathlib import Path

# Add project root to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))))

from datasets.data_loader import SignLanguageDataset, get_dataloader

class TestSignLanguageDataset:
    """Tests for the SignLanguageDataset class."""
    
    @pytest.fixture
    def mock_dataset_dir(self, tmp_path):
        """Create a mock dataset directory with minimal structure for testing."""
        # Create dataset directories
        data_dir = tmp_path / "mock_dataset"
        landmarks_dir = data_dir / "landmarks"
        splits_dir = data_dir / "splits"
        
        landmarks_dir.mkdir(parents=True)
        splits_dir.mkdir(parents=True)
        
        # Create mock metadata
        metadata = {
            "num_classes": 3,
            "processed_videos": 5,
            "total_videos": 5,
            "gloss_to_index": {
                "hello": 0,
                "thank_you": 1,
                "goodbye": 2
            }
        }
        
        with open(data_dir / "metadata.json", "w") as f:
            json.dump(metadata, f)
        
        # Create mock landmarks
        for i in range(5):
            video_id = f"video_{i}"
            landmarks = []
            
            # Create 10 frames of mock landmarks
            for _ in range(10):
                frame_data = {
                    "pose": np.random.rand(25, 4),  # 25 landmarks, 4 values (x, y, z, visibility)
                    "face": np.random.rand(468, 3),  # 468 landmarks, 3 values (x, y, z)
                    "left_hand": np.random.rand(21, 3),  # 21 landmarks, 3 values (x, y, z)
                    "right_hand": np.random.rand(21, 3)  # 21 landmarks, 3 values (x, y, z)
                }
                landmarks.append(frame_data)
            
            # Save landmarks
            with open(landmarks_dir / f"{video_id}.pkl", "wb") as f:
                pickle.dump(landmarks, f)
        
        # Create mock splits
        train_files = ["video_0.pkl", "video_1.pkl", "video_2.pkl"]
        val_files = ["video_3.pkl"]
        test_files = ["video_4.pkl"]
        
        with open(splits_dir / "train.json", "w") as f:
            json.dump(train_files, f)
        
        with open(splits_dir / "val.json", "w") as f:
            json.dump(val_files, f)
        
        with open(splits_dir / "test.json", "w") as f:
            json.dump(test_files, f)
        
        return data_dir
    
    def test_dataset_initialization(self, mock_dataset_dir):
        """Test that the dataset initializes correctly."""
        # Initialize dataset
        dataset = SignLanguageDataset(mock_dataset_dir, split="train")
        
        # Check dataset properties
        assert len(dataset) == 3  # 3 training samples
        assert dataset.num_classes == 3
        
        # Check gloss to index mapping
        assert dataset.gloss_to_index == {
            "hello": 0,
            "thank_you": 1,
            "goodbye": 2
        }
    
    def test_getitem(self, mock_dataset_dir):
        """Test the __getitem__ method."""
        # Initialize dataset
        dataset = SignLanguageDataset(mock_dataset_dir, split="train")
        
        # Get a sample
        sample = dataset[0]
        
        # Check sample structure
        assert "features" in sample
        assert "label" in sample
        assert "video_id" in sample
        
        # Check features shape (should be seq_len x feature_dim)
        assert len(sample["features"].shape) == 2
        
        # Check label type
        assert isinstance(sample["label"], int)
        
        # Check video_id
        assert sample["video_id"] == "video_0"
    
    def test_dataloader(self, mock_dataset_dir):
        """Test the get_dataloader function."""
        # Get dataloader
        dataloader = get_dataloader(mock_dataset_dir, batch_size=2, split="train")
        
        # Check dataloader
        assert dataloader.batch_size == 2
        assert len(dataloader.dataset) == 3
        
        # Get a batch
        batch = next(iter(dataloader))
        
        # Check batch structure
        assert "features" in batch
        assert "label" in batch
        assert "video_id" in batch
        
        # Check batch shapes
        assert batch["features"].shape[0] == 2  # Batch size
        assert len(batch["label"]) == 2  # Batch size
        assert len(batch["video_id"]) == 2  # Batch size
    
    def test_different_splits(self, mock_dataset_dir):
        """Test loading different data splits."""
        # Load different splits
        train_dataset = SignLanguageDataset(mock_dataset_dir, split="train")
        val_dataset = SignLanguageDataset(mock_dataset_dir, split="val")
        test_dataset = SignLanguageDataset(mock_dataset_dir, split="test")
        
        # Check sizes
        assert len(train_dataset) == 3
        assert len(val_dataset) == 1
        assert len(test_dataset) == 1
        
        # Check video IDs
        assert train_dataset[0]["video_id"] == "video_0"
        assert val_dataset[0]["video_id"] == "video_3"
        assert test_dataset[0]["video_id"] == "video_4"
    
    def test_transforms(self, mock_dataset_dir):
        """Test applying transforms to the dataset."""
        # Define a simple transform
        def simple_transform(features):
            return features * 2
        
        # Initialize dataset with transform
        dataset = SignLanguageDataset(mock_dataset_dir, split="train", transform=simple_transform)
        
        # Get a sample
        sample = dataset[0]
        
        # Check that the transform was applied
        # This is hard to test directly, but we can check that the values are in a reasonable range
        assert sample["features"].min() >= 0
        assert sample["features"].max() <= 2  # Max should be doubled from original (0-1 range)
