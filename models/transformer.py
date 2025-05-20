#!/usr/bin/env python
# models/transformer.py
# Transformer-based model for sign language recognition

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import numpy as np

class PositionalEncoding(nn.Module):
    """
    Positional encoding layer for Transformer.
    
    This layer adds positional information to the input embeddings.
    """
    def __init__(self, d_model, max_seq_length=5000):
        """
        Initialize the positional encoding layer.
        
        Args:
            d_model: Dimension of the model
            max_seq_length: Maximum sequence length
        """
        super(PositionalEncoding, self).__init__()
        
        # Create a positional encoding matrix
        pe = torch.zeros(max_seq_length, d_model)
        position = torch.arange(0, max_seq_length, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        
        pe = pe.unsqueeze(0)
        
        # Register as a buffer (not a parameter)
        self.register_buffer('pe', pe)
    
    def forward(self, x):
        """
        Forward pass of the positional encoding layer.
        
        Args:
            x: Input tensor of shape [batch_size, seq_length, d_model]
        
        Returns:
            Output tensor with positional encoding added
        """
        x = x + self.pe[:, :x.size(1), :]
        return x

class TransformerModel(nn.Module):
    """
    Transformer-based model for sign language recognition.
    
    This model uses a transformer encoder to process sequences of features.
    """
    def __init__(self, 
                 input_dim=258,  # (21 landmarks * 3 coords * 2 hands) + (25 landmarks * 3 coords for pose)
                 d_model=256, 
                 nhead=8, 
                 num_encoder_layers=4, 
                 dim_feedforward=2048, 
                 dropout=0.1, 
                 num_classes=100,
                 max_seq_length=100):
        """
        Initialize the transformer model.
        
        Args:
            input_dim: Dimension of input features
            d_model: Dimension of the model
            nhead: Number of heads in multi-head attention
            num_encoder_layers: Number of encoder layers
            dim_feedforward: Dimension of the feedforward network
            dropout: Dropout probability
            num_classes: Number of output classes
            max_seq_length: Maximum sequence length
        """
        super(TransformerModel, self).__init__()
        
        self.input_dim = input_dim
        self.d_model = d_model
        self.nhead = nhead
        self.num_encoder_layers = num_encoder_layers
        self.dim_feedforward = dim_feedforward
        self.dropout_p = dropout
        self.num_classes = num_classes
        self.max_seq_length = max_seq_length
        
        # Input projection
        self.input_projection = nn.Linear(input_dim, d_model)
        
        # Positional encoding
        self.positional_encoding = PositionalEncoding(d_model, max_seq_length)
        
        # Transformer encoder
        encoder_layers = nn.TransformerEncoderLayer(
            d_model=d_model, 
            nhead=nhead, 
            dim_feedforward=dim_feedforward, 
            dropout=dropout,
            batch_first=True
        )
        self.transformer_encoder = nn.TransformerEncoder(
            encoder_layers, 
            num_layers=num_encoder_layers
        )
        
        # Output layer
        self.output_layer = nn.Linear(d_model, num_classes)
        
        # Dropout
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, x, src_key_padding_mask=None):
        """
        Forward pass of the transformer model.
        
        Args:
            x: Input tensor of shape [batch_size, seq_length, input_dim]
            src_key_padding_mask: Mask for padding tokens (optional)
        
        Returns:
            Output tensor of shape [batch_size, num_classes]
        """
        # Project input to d_model dimensions
        x = self.input_projection(x)
        
        # Add positional encoding
        x = self.positional_encoding(x)
        
        # Apply dropout
        x = self.dropout(x)
        
        # Transformer encoder
        x = self.transformer_encoder(x, src_key_padding_mask=src_key_padding_mask)
        
        # Global average pooling
        x = torch.mean(x, dim=1)
        
        # Output layer
        x = self.output_layer(x)
        
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
            'd_model': self.d_model,
            'nhead': self.nhead,
            'num_encoder_layers': self.num_encoder_layers,
            'dim_feedforward': self.dim_feedforward,
            'dropout': self.dropout_p,
            'num_classes': self.num_classes,
            'max_seq_length': self.max_seq_length
        }
        
        checkpoint = {
            'model_args': model_args,
            'state_dict': self.state_dict()
        }
        
        torch.save(checkpoint, checkpoint_path)

class TransformerEncoderDecoderModel(nn.Module):
    """
    Transformer Encoder-Decoder model for sign language translation.
    
    This model uses a transformer encoder-decoder architecture to translate
    sign language to text.
    """
    def __init__(self, 
                 input_dim=258,  # (21 landmarks * 3 coords * 2 hands) + (25 landmarks * 3 coords for pose)
                 d_model=256, 
                 nhead=8, 
                 num_encoder_layers=4, 
                 num_decoder_layers=4,
                 dim_feedforward=2048, 
                 dropout=0.1, 
                 vocab_size=5000,
                 max_seq_length=100):
        """
        Initialize the transformer encoder-decoder model.
        
        Args:
            input_dim: Dimension of input features
            d_model: Dimension of the model
            nhead: Number of heads in multi-head attention
            num_encoder_layers: Number of encoder layers
            num_decoder_layers: Number of decoder layers
            dim_feedforward: Dimension of the feedforward network
            dropout: Dropout probability
            vocab_size: Size of the vocabulary (including special tokens)
            max_seq_length: Maximum sequence length
        """
        super(TransformerEncoderDecoderModel, self).__init__()
        
        self.input_dim = input_dim
        self.d_model = d_model
        self.nhead = nhead
        self.num_encoder_layers = num_encoder_layers
        self.num_decoder_layers = num_decoder_layers
        self.dim_feedforward = dim_feedforward
        self.dropout_p = dropout
        self.vocab_size = vocab_size
        self.max_seq_length = max_seq_length
        
        # Input projection
        self.input_projection = nn.Linear(input_dim, d_model)
        
        # Positional encoding
        self.src_positional_encoding = PositionalEncoding(d_model, max_seq_length)
        self.tgt_positional_encoding = PositionalEncoding(d_model, max_seq_length)
        
        # Embedding layer for target tokens
        self.tgt_embedding = nn.Embedding(vocab_size, d_model)
        
        # Transformer encoder-decoder
        encoder_layers = nn.TransformerEncoderLayer(
            d_model=d_model, 
            nhead=nhead, 
            dim_feedforward=dim_feedforward, 
            dropout=dropout,
            batch_first=True
        )
        self.transformer_encoder = nn.TransformerEncoder(
            encoder_layers, 
            num_layers=num_encoder_layers
        )
        
        decoder_layers = nn.TransformerDecoderLayer(
            d_model=d_model, 
            nhead=nhead, 
            dim_feedforward=dim_feedforward, 
            dropout=dropout,
            batch_first=True
        )
        self.transformer_decoder = nn.TransformerDecoder(
            decoder_layers, 
            num_layers=num_decoder_layers
        )
        
        # Output layer
        self.output_layer = nn.Linear(d_model, vocab_size)
        
        # Dropout
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, src, tgt, src_key_padding_mask=None, tgt_key_padding_mask=None, tgt_mask=None):
        """
        Forward pass of the transformer encoder-decoder model.
        
        Args:
            src: Source sequence of shape [batch_size, src_seq_length, input_dim]
            tgt: Target sequence of shape [batch_size, tgt_seq_length]
            src_key_padding_mask: Mask for source padding tokens (optional)
            tgt_key_padding_mask: Mask for target padding tokens (optional)
            tgt_mask: Mask to prevent attention to future tokens (optional)
        
        Returns:
            Output tensor of shape [batch_size, tgt_seq_length, vocab_size]
        """
        # Create target mask if not provided
        if tgt_mask is None:
            tgt_seq_length = tgt.size(1)
            tgt_mask = self._generate_square_subsequent_mask(tgt_seq_length).to(src.device)
        
        # Project input to d_model dimensions
        src = self.input_projection(src)
        
        # Add positional encoding to source
        src = self.src_positional_encoding(src)
        
        # Apply dropout to source
        src = self.dropout(src)
        
        # Transformer encoder
        memory = self.transformer_encoder(src, src_key_padding_mask=src_key_padding_mask)
        
        # Embed target tokens
        tgt = self.tgt_embedding(tgt)
        
        # Add positional encoding to target
        tgt = self.tgt_positional_encoding(tgt)
        
        # Apply dropout to target
        tgt = self.dropout(tgt)
        
        # Transformer decoder
        output = self.transformer_decoder(
            tgt, 
            memory, 
            tgt_mask=tgt_mask, 
            tgt_key_padding_mask=tgt_key_padding_mask,
            memory_key_padding_mask=src_key_padding_mask
        )
        
        # Output layer
        output = self.output_layer(output)
        
        return output
    
    def _generate_square_subsequent_mask(self, size):
        """
        Generate a square mask for the sequence.
        
        The mask ensures that the predictions for position i can depend only on the known outputs at positions less than i.
        
        Args:
            size: Sequence length
        
        Returns:
            Mask tensor of shape [size, size]
        """
        mask = (torch.triu(torch.ones(size, size)) == 1).transpose(0, 1)
        mask = mask.float().masked_fill(mask == 0, float('-inf')).masked_fill(mask == 1, float(0.0))
        return mask
    
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
            'd_model': self.d_model,
            'nhead': self.nhead,
            'num_encoder_layers': self.num_encoder_layers,
            'num_decoder_layers': self.num_decoder_layers,
            'dim_feedforward': self.dim_feedforward,
            'dropout': self.dropout_p,
            'vocab_size': self.vocab_size,
            'max_seq_length': self.max_seq_length
        }
        
        checkpoint = {
            'model_args': model_args,
            'state_dict': self.state_dict()
        }
        
        torch.save(checkpoint, checkpoint_path)

if __name__ == "__main__":
    # Test the models
    
    # Test TransformerModel
    batch_size = 10
    seq_len = 30
    input_dim = 258
    num_classes = 100
    
    # Create a random input tensor
    x = torch.randn(batch_size, seq_len, input_dim)
    
    # Create the model
    model = TransformerModel(input_dim=input_dim, num_classes=num_classes)
    
    # Forward pass
    output = model(x)
    
    print(f"Input shape: {x.shape}")
    print(f"Output shape: {output.shape}")
    
    # Test TransformerEncoderDecoderModel
    tgt_seq_len = 20
    vocab_size = 5000
    
    # Create random input tensors
    src = torch.randn(batch_size, seq_len, input_dim)
    tgt = torch.randint(0, vocab_size, (batch_size, tgt_seq_len))
    
    # Create the model
    enc_dec_model = TransformerEncoderDecoderModel(input_dim=input_dim, vocab_size=vocab_size)
    
    # Forward pass
    output = enc_dec_model(src, tgt)
    
    print(f"Source shape: {src.shape}")
    print(f"Target shape: {tgt.shape}")
    print(f"Output shape: {output.shape}")
