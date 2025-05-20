#!/usr/bin/env python
# tests/conftest.py
# pytest configuration file with fixtures

import os
import sys
import pytest
import torch
import numpy as np
import json
import pickle
from pathlib import Path

# Add project root to path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Common test fixtures
@pytest.fixture
def random_seed():
    """Set a fixed random seed for reproducibility."""
    seed = 42
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    return seed

@pytest.fixture
def device():
    """Get compute device."""
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")

@pytest.fixture
def sample_landmarks():
    """Generate sample landmarks for a single frame."""
    return {
        "pose": np.random.rand(25, 4),  # 25 landmarks, 4 values (x, y, z, visibility)
        "face": np.random.rand(468, 3),  # 468 landmarks, 3 values (x, y, z)
        "left_hand": np.random.rand(21, 3),  # 21 landmarks, 3 values (x, y, z)
        "right_hand": np.random.rand(21, 3)  # 21 landmarks, 3 values (x, y, z)
    }

@pytest.fixture
def sample_video_landmarks(sample_landmarks):
    """Generate sample landmarks for a video sequence."""
    return [sample_landmarks for _ in range(30)]  # 30 frames

@pytest.fixture
def sample_batch(random_seed):
    """Generate a sample batch for model testing."""
    batch_size = 4
    seq_len = 20
    input_dim = 258  # (21 landmarks * 3 coords * 2 hands) + (25 landmarks * 3 coords for pose)
    
    features = torch.randn(batch_size, seq_len, input_dim)
    labels = torch.randint(0, 10, (batch_size,))
    
    return {
        "features": features,
        "label": labels,
        "video_id": [f"video_{i}" for i in range(batch_size)]
    }

@pytest.fixture
def sample_image_batch(random_seed):
    """Generate a sample image batch for CNN model testing."""
    batch_size = 4
    seq_len = 10
    channels = 3
    height = 224
    width = 224
    
    images = torch.randn(batch_size, seq_len, channels, height, width)
    labels = torch.randint(0, 10, (batch_size,))
    
    return {
        "features": images,
        "label": labels,
        "video_id": [f"video_{i}" for i in range(batch_size)]
    }

@pytest.fixture
def mock_model_checkpoint(tmp_path, random_seed):
    """Create a mock model checkpoint for testing."""
    checkpoint_dir = tmp_path / "checkpoints"
    checkpoint_dir.mkdir(exist_ok=True)
    
    checkpoint_path = checkpoint_dir / "model.pth"
    
    # Create a simple checkpoint structure
    checkpoint = {
        "model_args": {
            "input_dim": 258,
            "num_classes": 10,
            "hidden_dim": 128,
            "lstm_hidden_size": 256,
            "lstm_num_layers": 2,
            "dropout": 0.5,
            "bidirectional": True
        },
        "state_dict": {
            "layer1.weight": torch.randn(128, 258),
            "layer1.bias": torch.randn(128),
            "layer2.weight": torch.randn(256, 128),
            "layer2.bias": torch.randn(256),
            "output.weight": torch.randn(10, 256),
            "output.bias": torch.randn(10)
        }
    }
    
    torch.save(checkpoint, checkpoint_path)
    
    return checkpoint_path

@pytest.fixture
def mock_dataset_dir(tmp_path, random_seed):
    """Create a mock dataset directory structure for testing."""
    data_dir = tmp_path / "dataset"
    raw_dir = data_dir / "raw"
    processed_dir = data_dir / "processed"
    features_dir = data_dir / "features"
    
    # Create directories
    raw_dir.mkdir(parents=True)
    processed_dir.mkdir(parents=True)
    features_dir.mkdir(parents=True)
    
    return data_dir
