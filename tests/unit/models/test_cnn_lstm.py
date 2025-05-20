#!/usr/bin/env python
# tests/unit/models/test_cnn_lstm.py
# Unit tests for CNN-LSTM model

import os
import sys
import pytest
import torch
import numpy as np

# Add project root to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))))

from models.cnn_lstm import CNNLSTM, LandmarkCNNLSTM

class TestCNNLSTM:
    """Tests for the CNNLSTM model."""
    
    def test_model_initialization(self):
        """Test that the model initializes correctly with default parameters."""
        model = CNNLSTM(num_classes=10)
        
        # Check model properties
        assert model.num_classes == 10
        assert model.input_channels == 3
        assert model.lstm_hidden_size == 256
        assert model.lstm_num_layers == 2
        assert model.bidirectional == True
    
    def test_forward_pass(self):
        """Test the forward pass of the model."""
        # Create a model
        model = CNNLSTM(num_classes=10)
        
        # Create a random input tensor
        batch_size = 5
        seq_len = 10
        channels = 3
        height = 224
        width = 224
        x = torch.randn(batch_size, seq_len, channels, height, width)
        
        # Perform forward pass
        output = model(x)
        
        # Check output shape
        assert output.shape == (batch_size, 10)
        
        # Check output is not all zeros or NaNs
        assert not torch.isnan(output).any()
        assert not (output == 0).all()
    
    def test_save_and_load(self, tmp_path):
        """Test saving and loading the model."""
        # Create a model
        model = CNNLSTM(num_classes=10)
        
        # Save the model
        checkpoint_path = os.path.join(tmp_path, "model.pth")
        model.save_checkpoint(checkpoint_path)
        
        # Check that the checkpoint file exists
        assert os.path.exists(checkpoint_path)
        
        # Load the model
        loaded_model = CNNLSTM.load_from_checkpoint(checkpoint_path)
        
        # Check that the loaded model has the same parameters
        assert loaded_model.num_classes == model.num_classes
        assert loaded_model.input_channels == model.input_channels
        assert loaded_model.lstm_hidden_size == model.lstm_hidden_size
        assert loaded_model.lstm_num_layers == model.lstm_num_layers
        assert loaded_model.bidirectional == model.bidirectional
        
        # Check that the model parameters are the same
        for p1, p2 in zip(model.parameters(), loaded_model.parameters()):
            assert torch.allclose(p1, p2)
    
    def test_extract_features(self):
        """Test the feature extraction method."""
        # Create a model
        model = CNNLSTM(num_classes=10)
        
        # Create a random input tensor
        batch_size = 5
        seq_len = 10
        channels = 3
        height = 224
        width = 224
        x = torch.randn(batch_size, seq_len, channels, height, width)
        
        # Extract features
        features = model.extract_features(x)
        
        # Check features shape (should be batch_size x lstm_hidden_size*2 for bidirectional)
        expected_dim = model.lstm_hidden_size * (2 if model.bidirectional else 1)
        assert features.shape == (batch_size, expected_dim)


class TestLandmarkCNNLSTM:
    """Tests for the LandmarkCNNLSTM model."""
    
    def test_model_initialization(self):
        """Test that the model initializes correctly with default parameters."""
        input_dim = 258  # (21 landmarks * 3 coords * 2 hands) + (25 landmarks * 3 coords for pose)
        model = LandmarkCNNLSTM(input_dim=input_dim, num_classes=10)
        
        # Check model properties
        assert model.num_classes == 10
        assert model.input_dim == input_dim
        assert model.lstm_hidden_size == 512
        assert model.lstm_num_layers == 2
        assert model.bidirectional == True
    
    def test_forward_pass(self):
        """Test the forward pass of the model."""
        # Create a model
        input_dim = 258
        model = LandmarkCNNLSTM(input_dim=input_dim, num_classes=10)
        
        # Create a random input tensor
        batch_size = 5
        seq_len = 20
        x = torch.randn(batch_size, seq_len, input_dim)
        
        # Perform forward pass
        output = model(x)
        
        # Check output shape
        assert output.shape == (batch_size, 10)
        
        # Check output is not all zeros or NaNs
        assert not torch.isnan(output).any()
        assert not (output == 0).all()
    
    def test_variable_sequence_length(self):
        """Test the model with variable sequence lengths."""
        # Create a model
        input_dim = 258
        model = LandmarkCNNLSTM(input_dim=input_dim, num_classes=10)
        
        # Create a random input tensor
        batch_size = 5
        max_seq_len = 20
        x = torch.randn(batch_size, max_seq_len, input_dim)
        
        # Create sequence lengths
        lengths = torch.randint(5, max_seq_len + 1, (batch_size,))
        
        # Perform forward pass
        output = model(x, lengths)
        
        # Check output shape
        assert output.shape == (batch_size, 10)
        
        # Check output is not all zeros or NaNs
        assert not torch.isnan(output).any()
        assert not (output == 0).all()
