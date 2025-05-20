#!/usr/bin/env python
# models/cnn_lstm.py
# CNN+LSTM hybrid model for sign language recognition

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence

class CNNLSTM(nn.Module):
    """
    CNN+LSTM hybrid model for sign language recognition.
    
    This model uses a CNN to extract spatial features from each frame,
    and an LSTM to capture temporal dependencies across frames.
    """
    def __init__(self, 
                 input_channels=3, 
                 num_classes=100, 
                 cnn_output_size=128, 
                 lstm_hidden_size=256, 
                 lstm_num_layers=2, 
                 dropout=0.5,
                 bidirectional=True):
        """
        Initialize the CNN+LSTM model.
        
        Args:
            input_channels: Number of input channels (3 for RGB videos)
            num_classes: Number of output classes
            cnn_output_size: Size of CNN output features
            lstm_hidden_size: Size of LSTM hidden state
            lstm_num_layers: Number of LSTM layers
            dropout: Dropout probability
            bidirectional: Whether to use bidirectional LSTM
        """
        super(CNNLSTM, self).__init__()
        
        self.input_channels = input_channels
        self.num_classes = num_classes
        self.cnn_output_size = cnn_output_size
        self.lstm_hidden_size = lstm_hidden_size
        self.lstm_num_layers = lstm_num_layers
        self.dropout_p = dropout
        self.bidirectional = bidirectional
        
        # CNN layers for feature extraction
        self.conv1 = nn.Conv2d(input_channels, 64, kernel_size=3, stride=1, padding=1)
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.conv2 = nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1)
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.conv3 = nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1)
        self.pool3 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.conv4 = nn.Conv2d(256, 512, kernel_size=3, stride=1, padding=1)
        self.pool4 = nn.MaxPool2d(kernel_size=2, stride=2)
        
        # Adaptive pooling to ensure fixed size output regardless of input dimensions
        self.adaptive_pool = nn.AdaptiveAvgPool2d((1, 1))
        
        # Fully connected layer after CNN
        self.fc_cnn = nn.Linear(512, cnn_output_size)
        
        # LSTM for sequence modeling
        self.lstm = nn.LSTM(
            input_size=cnn_output_size,
            hidden_size=lstm_hidden_size,
            num_layers=lstm_num_layers,
            batch_first=True,
            dropout=dropout if lstm_num_layers > 1 else 0,
            bidirectional=bidirectional
        )
        
        # Dropout layer
        self.dropout = nn.Dropout(dropout)
        
        # Fully connected layer for classification
        lstm_output_size = lstm_hidden_size * 2 if bidirectional else lstm_hidden_size
        self.fc = nn.Linear(lstm_output_size, num_classes)
    
    def forward(self, x, lengths=None):
        """
        Forward pass of the CNN+LSTM model.
        
        Args:
            x: Input tensor of shape [batch_size, sequence_length, channels, height, width]
            lengths: Sequence lengths for packed sequence (optional)
        
        Returns:
            Output tensor of shape [batch_size, num_classes]
        """
        batch_size, seq_len = x.size(0), x.size(1)
        
        # Reshape for CNN: [batch_size * seq_len, channels, height, width]
        x = x.view(-1, self.input_channels, x.size(3), x.size(4))
        
        # CNN feature extraction
        x = F.relu(self.conv1(x))
        x = self.pool1(x)
        x = F.relu(self.conv2(x))
        x = self.pool2(x)
        x = F.relu(self.conv3(x))
        x = self.pool3(x)
        x = F.relu(self.conv4(x))
        x = self.pool4(x)
        
        # Adaptive pooling to get fixed size
        x = self.adaptive_pool(x)
        
        # Flatten the output
        x = x.view(-1, 512)
        
        # Fully connected layer
        x = F.relu(self.fc_cnn(x))
        
        # Reshape back to sequence: [batch_size, seq_len, cnn_output_size]
        x = x.view(batch_size, seq_len, self.cnn_output_size)
        
        # LSTM sequence modeling
        if lengths is not None:
            # Pack the sequence
            x = pack_padded_sequence(x, lengths, batch_first=True, enforce_sorted=False)
        
        # Process with LSTM
        x, _ = self.lstm(x)
        
        if lengths is not None:
            # Unpack the sequence
            x, _ = pad_packed_sequence(x, batch_first=True)
        
        # Take the last output for each sequence
        if lengths is not None:
            # Get the last output for each sequence based on its length
            x = torch.stack([x[i, length - 1] for i, length in enumerate(lengths)])
        else:
            # Take the last time step if lengths are not provided
            x = x[:, -1]
        
        # Dropout
        x = self.dropout(x)
        
        # Classification
        x = self.fc(x)
        
        return x
    
    def extract_features(self, x, lengths=None):
        """
        Extract features from the model (useful for transfer learning).
        
        Args:
            x: Input tensor of shape [batch_size, sequence_length, channels, height, width]
            lengths: Sequence lengths for packed sequence (optional)
        
        Returns:
            Features tensor of shape [batch_size, lstm_hidden_size * (2 if bidirectional else 1)]
        """
        batch_size, seq_len = x.size(0), x.size(1)
        
        # Reshape for CNN: [batch_size * seq_len, channels, height, width]
        x = x.view(-1, self.input_channels, x.size(3), x.size(4))
        
        # CNN feature extraction
        x = F.relu(self.conv1(x))
        x = self.pool1(x)
        x = F.relu(self.conv2(x))
        x = self.pool2(x)
        x = F.relu(self.conv3(x))
        x = self.pool3(x)
        x = F.relu(self.conv4(x))
        x = self.pool4(x)
        
        # Adaptive pooling to get fixed size
        x = self.adaptive_pool(x)
        
        # Flatten the output
        x = x.view(-1, 512)
        
        # Fully connected layer
        x = F.relu(self.fc_cnn(x))
        
        # Reshape back to sequence: [batch_size, seq_len, cnn_output_size]
        x = x.view(batch_size, seq_len, self.cnn_output_size)
        
        # LSTM sequence modeling
        if lengths is not None:
            # Pack the sequence
            x = pack_padded_sequence(x, lengths, batch_first=True, enforce_sorted=False)
        
        # Process with LSTM
        x, _ = self.lstm(x)
        
        if lengths is not None:
            # Unpack the sequence
            x, _ = pad_packed_sequence(x, batch_first=True)
        
        # Take the last output for each sequence
        if lengths is not None:
            # Get the last output for each sequence based on its length
            x = torch.stack([x[i, length - 1] for i, length in enumerate(lengths)])
        else:
            # Take the last time step if lengths are not provided
            x = x[:, -1]
        
        return x
    
    @classmethod
    def load_from_checkpoint(cls, checkpoint_path):
        """
        Load model from checkpoint.
        
        Args:
            checkpoint_path: Path to the checkpoint file
        
        Returns:
            Loaded model
        """
        checkpoint = torch.load(checkpoint_path, map_location=torch.device('cpu'))
        model_args = checkpoint['model_args']
        model = cls(**model_args)
        model.load_state_dict(checkpoint['state_dict'])
        return model
    
    def save_checkpoint(self, checkpoint_path):
        """
        Save model checkpoint.
        
        Args:
            checkpoint_path: Path to save the checkpoint file
        """
        model_args = {
            'input_channels': self.input_channels,
            'num_classes': self.num_classes,
            'cnn_output_size': self.cnn_output_size,
            'lstm_hidden_size': self.lstm_hidden_size,
            'lstm_num_layers': self.lstm_num_layers,
            'dropout': self.dropout_p,
            'bidirectional': self.bidirectional
        }
        
        checkpoint = {
            'model_args': model_args,
            'state_dict': self.state_dict()
        }
        
        torch.save(checkpoint, checkpoint_path)

class LandmarkCNNLSTM(nn.Module):
    """
    CNN+LSTM model for landmark-based sign language recognition.
    
    This model takes preprocessed landmarks as input instead of raw images.
    """
    def __init__(self, 
                 input_dim=258,  # (21 landmarks * 3 coords * 2 hands) + (25 landmarks * 3 coords for pose)
                 num_classes=100, 
                 hidden_dim=256, 
                 lstm_hidden_size=512, 
                 lstm_num_layers=2, 
                 dropout=0.5,
                 bidirectional=True):
        """
        Initialize the Landmark CNN+LSTM model.
        
        Args:
            input_dim: Dimension of input features (landmarks)
            num_classes: Number of output classes
            hidden_dim: Dimension of hidden layer
            lstm_hidden_size: Size of LSTM hidden state
            lstm_num_layers: Number of LSTM layers
            dropout: Dropout probability
            bidirectional: Whether to use bidirectional LSTM
        """
        super(LandmarkCNNLSTM, self).__init__()
        
        self.input_dim = input_dim
        self.num_classes = num_classes
        self.hidden_dim = hidden_dim
        self.lstm_hidden_size = lstm_hidden_size
        self.lstm_num_layers = lstm_num_layers
        self.dropout_p = dropout
        self.bidirectional = bidirectional
        
        # Feature extraction layers
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.bn1 = nn.BatchNorm1d(hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.bn2 = nn.BatchNorm1d(hidden_dim)
        
        # LSTM for sequence modeling
        self.lstm = nn.LSTM(
            input_size=hidden_dim,
            hidden_size=lstm_hidden_size,
            num_layers=lstm_num_layers,
            batch_first=True,
            dropout=dropout if lstm_num_layers > 1 else 0,
            bidirectional=bidirectional
        )
        
        # Dropout layer
        self.dropout = nn.Dropout(dropout)
        
        # Fully connected layer for classification
        lstm_output_size = lstm_hidden_size * 2 if bidirectional else lstm_hidden_size
        self.fc = nn.Linear(lstm_output_size, num_classes)
    
    def forward(self, x, lengths=None):
        """
        Forward pass of the Landmark CNN+LSTM model.
        
        Args:
            x: Input tensor of shape [batch_size, sequence_length, input_dim]
            lengths: Sequence lengths for packed sequence (optional)
        
        Returns:
            Output tensor of shape [batch_size, num_classes]
        """
        batch_size, seq_len = x.size(0), x.size(1)
        
        # Reshape for batch processing: [batch_size * seq_len, input_dim]
        x = x.view(-1, self.input_dim)
        
        # Feature extraction
        x = F.relu(self.bn1(self.fc1(x)))
        x = F.relu(self.bn2(self.fc2(x)))
        
        # Reshape back to sequence: [batch_size, seq_len, hidden_dim]
        x = x.view(batch_size, seq_len, self.hidden_dim)
        
        # LSTM sequence modeling
        if lengths is not None:
            # Pack the sequence
            x = pack_padded_sequence(x, lengths, batch_first=True, enforce_sorted=False)
        
        # Process with LSTM
        x, _ = self.lstm(x)
        
        if lengths is not None:
            # Unpack the sequence
            x, _ = pad_packed_sequence(x, batch_first=True)
        
        # Take the last output for each sequence
        if lengths is not None:
            # Get the last output for each sequence based on its length
            x = torch.stack([x[i, length - 1] for i, length in enumerate(lengths)])
        else:
            # Take the last time step if lengths are not provided
            x = x[:, -1]
        
        # Dropout
        x = self.dropout(x)
        
        # Classification
        x = self.fc(x)
        
        return x
    
    @classmethod
    def load_from_checkpoint(cls, checkpoint_path):
        """
        Load model from checkpoint.
        
        Args:
            checkpoint_path: Path to the checkpoint file
        
        Returns:
            Loaded model
        """
        checkpoint = torch.load(checkpoint_path, map_location=torch.device('cpu'))
        model_args = checkpoint['model_args']
        model = cls(**model_args)
        model.load_state_dict(checkpoint['state_dict'])
        return model
    
    def save_checkpoint(self, checkpoint_path):
        """
        Save model checkpoint.
        
        Args:
            checkpoint_path: Path to save the checkpoint file
        """
        model_args = {
            'input_dim': self.input_dim,
            'num_classes': self.num_classes,
            'hidden_dim': self.hidden_dim,
            'lstm_hidden_size': self.lstm_hidden_size,
            'lstm_num_layers': self.lstm_num_layers,
            'dropout': self.dropout_p,
            'bidirectional': self.bidirectional
        }
        
        checkpoint = {
            'model_args': model_args,
            'state_dict': self.state_dict()
        }
        
        torch.save(checkpoint, checkpoint_path)

if __name__ == "__main__":
    # Test the models
    
    # Test CNNLSTM model
    batch_size = 10
    seq_len = 30
    channels = 3
    height = 224
    width = 224
    num_classes = 100
    
    # Create a random input tensor
    x = torch.randn(batch_size, seq_len, channels, height, width)
    
    # Create the model
    model = CNNLSTM(input_channels=channels, num_classes=num_classes)
    
    # Forward pass
    output = model(x)
    
    print(f"Input shape: {x.shape}")
    print(f"Output shape: {output.shape}")
    
    # Test LandmarkCNNLSTM model
    input_dim = 258  # (21 landmarks * 3 coords * 2 hands) + (25 landmarks * 3 coords for pose)
    
    # Create a random input tensor
    x = torch.randn(batch_size, seq_len, input_dim)
    
    # Create the model
    landmark_model = LandmarkCNNLSTM(input_dim=input_dim, num_classes=num_classes)
    
    # Forward pass
    output = landmark_model(x)
    
    print(f"Landmark input shape: {x.shape}")
    print(f"Landmark output shape: {output.shape}")
