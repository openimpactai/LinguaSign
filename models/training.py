#!/usr/bin/env python
# models/training.py
# Training script for sign language recognition models

import os
import argparse
import json
import time
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
import sys

# Add parent directory to path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from datasets.data_loader import SignLanguageDataset, get_dataloader
from models.cnn_lstm import CNNLSTM, LandmarkCNNLSTM
from models.mediapipe_ml import MediaPipeML, MediaPipeMLSequence
from models.transformer import TransformerModel, TransformerEncoderDecoderModel

def train_epoch(model, dataloader, criterion, optimizer, device):
    """
    Train the model for one epoch.
    
    Args:
        model: Model to train
        dataloader: DataLoader for training data
        criterion: Loss function
        optimizer: Optimizer
        device: Device to train on
    
    Returns:
        Average loss for the epoch
    """
    model.train()
    total_loss = 0.0
    correct = 0
    total = 0
    
    for batch in dataloader:
        # Get data and move to device
        features = batch['features'].to(device)
        labels = batch['label'].to(device)
        
        # Zero the parameter gradients
        optimizer.zero_grad()
        
        # Forward pass
        outputs = model(features)
        
        # Compute loss
        loss = criterion(outputs, labels)
        
        # Backward pass and optimize
        loss.backward()
        optimizer.step()
        
        # Update statistics
        total_loss += loss.item() * features.size(0)
        
        # Compute accuracy
        _, predicted = torch.max(outputs, 1)
        correct += (predicted == labels).sum().item()
        total += labels.size(0)
    
    epoch_loss = total_loss / len(dataloader.dataset)
    epoch_acc = correct / total
    
    return epoch_loss, epoch_acc

def validate(model, dataloader, criterion, device):
    """
    Validate the model.
    
    Args:
        model: Model to validate
        dataloader: DataLoader for validation data
        criterion: Loss function
        device: Device to validate on
    
    Returns:
        Average loss and accuracy for the validation set
    """
    model.eval()
    total_loss = 0.0
    correct = 0
    total = 0
    
    with torch.no_grad():
        for batch in dataloader:
            # Get data and move to device
            features = batch['features'].to(device)
            labels = batch['label'].to(device)
            
            # Forward pass
            outputs = model(features)
            
            # Compute loss
            loss = criterion(outputs, labels)
            
            # Update statistics
            total_loss += loss.item() * features.size(0)
            
            # Compute accuracy
            _, predicted = torch.max(outputs, 1)
            correct += (predicted == labels).sum().item()
            total += labels.size(0)
    
    val_loss = total_loss / len(dataloader.dataset)
    val_acc = correct / total
    
    return val_loss, val_acc

def train_model(model, train_dataloader, val_dataloader, criterion, optimizer, device, num_epochs=100, patience=10, checkpoint_dir=None):
    """
    Train the model.
    
    Args:
        model: Model to train
        train_dataloader: DataLoader for training data
        val_dataloader: DataLoader for validation data
        criterion: Loss function
        optimizer: Optimizer
        device: Device to train on
        num_epochs: Number of epochs to train for
        patience: Number of epochs to wait for improvement before early stopping
        checkpoint_dir: Directory to save model checkpoints
    
    Returns:
        Dictionary with training history
    """
    # Initialize variables for early stopping
    best_val_loss = float('inf')
    best_val_acc = 0.0
    best_model_state_dict = None
    patience_counter = 0
    
    # Initialize history
    history = {
        'train_loss': [],
        'train_acc': [],
        'val_loss': [],
        'val_acc': []
    }
    
    # Create checkpoint directory if it doesn't exist
    if checkpoint_dir is not None:
        os.makedirs(checkpoint_dir, exist_ok=True)
    
    # Train the model
    for epoch in range(num_epochs):
        start_time = time.time()
        
        # Train for one epoch
        train_loss, train_acc = train_epoch(model, train_dataloader, criterion, optimizer, device)
        
        # Validate
        val_loss, val_acc = validate(model, val_dataloader, criterion, device)
        
        # Update history
        history['train_loss'].append(train_loss)
        history['train_acc'].append(train_acc)
        history['val_loss'].append(val_loss)
        history['val_acc'].append(val_acc)
        
        # Print progress
        epoch_time = time.time() - start_time
        print(f"Epoch {epoch+1}/{num_epochs} - {epoch_time:.2f}s - train_loss: {train_loss:.4f} - train_acc: {train_acc:.4f} - val_loss: {val_loss:.4f} - val_acc: {val_acc:.4f}")
        
        # Check if this is the best model
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_val_loss = val_loss
            best_model_state_dict = model.state_dict().copy()
            patience_counter = 0
            
            # Save checkpoint
            if checkpoint_dir is not None:
                checkpoint_path = os.path.join(checkpoint_dir, f"model_epoch{epoch+1}_val_acc{val_acc:.4f}.pth")
                if hasattr(model, 'save_checkpoint'):
                    model.save_checkpoint(checkpoint_path)
                else:
                    torch.save(model.state_dict(), checkpoint_path)
                print(f"Checkpoint saved to {checkpoint_path}")
        else:
            patience_counter += 1
            
            # Early stopping
            if patience_counter >= patience:
                print(f"Early stopping after {epoch+1} epochs")
                break
    
    # Load the best model
    if best_model_state_dict is not None:
        model.load_state_dict(best_model_state_dict)
    
    # Save final model
    if checkpoint_dir is not None:
        final_checkpoint_path = os.path.join(checkpoint_dir, "model_final.pth")
        if hasattr(model, 'save_checkpoint'):
            model.save_checkpoint(final_checkpoint_path)
        else:
            torch.save(model.state_dict(), final_checkpoint_path)
        print(f"Final model saved to {final_checkpoint_path}")
        
        # Save training history
        history_path = os.path.join(checkpoint_dir, "training_history.json")
        with open(history_path, 'w') as f:
            json.dump(history, f)
        print(f"Training history saved to {history_path}")
    
    return history, best_val_acc, best_val_loss

def create_model(model_type, input_dim, num_classes, device):
    """
    Create a model.
    
    Args:
        model_type: Type of model to create
        input_dim: Input dimension
        num_classes: Number of output classes
        device: Device to create the model on
    
    Returns:
        Created model
    """
    if model_type == 'cnn_lstm':
        model = CNNLSTM(input_channels=3, num_classes=num_classes)
    elif model_type == 'landmark_cnn_lstm':
        model = LandmarkCNNLSTM(input_dim=input_dim, num_classes=num_classes)
    elif model_type == 'mediapipe_ml':
        model = MediaPipeML(classifier_type='svc')
    elif model_type == 'mediapipe_ml_sequence':
        model = MediaPipeMLSequence(classifier_type='svc')
    elif model_type == 'transformer':
        model = TransformerModel(input_dim=input_dim, num_classes=num_classes)
    else:
        raise ValueError(f"Unknown model type: {model_type}")
    
    # Move model to device
    if model_type not in ['mediapipe_ml', 'mediapipe_ml_sequence']:
        model = model.to(device)
    
    return model

def main():
    """Main function for training a sign language recognition model."""
    parser = argparse.ArgumentParser(description="Train a sign language recognition model")
    parser.add_argument("--model_type", type=str, default="landmark_cnn_lstm", 
                        choices=["cnn_lstm", "landmark_cnn_lstm", "mediapipe_ml", "mediapipe_ml_sequence", "transformer"],
                        help="Type of model to train")
    parser.add_argument("--data_dir", type=str, required=True, 
                        help="Directory containing the processed dataset")
    parser.add_argument("--output_dir", type=str, default="./checkpoints", 
                        help="Directory to save model checkpoints")
    parser.add_argument("--batch_size", type=int, default=32, 
                        help="Batch size")
    parser.add_argument("--num_epochs", type=int, default=100, 
                        help="Number of epochs to train for")
    parser.add_argument("--learning_rate", type=float, default=0.001, 
                        help="Learning rate")
    parser.add_argument("--patience", type=int, default=10, 
                        help="Number of epochs to wait for improvement before early stopping")
    parser.add_argument("--seed", type=int, default=42, 
                        help="Random seed for reproducibility")
    
    args = parser.parse_args()
    
    # Set random seed for reproducibility
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    
    # Determine device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Create output directory if it doesn't exist
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Create model-specific output directory
    model_output_dir = os.path.join(args.output_dir, args.model_type)
    os.makedirs(model_output_dir, exist_ok=True)
    
    # Get dataloaders
    train_dataloader = get_dataloader(args.data_dir, batch_size=args.batch_size, split='train')
    val_dataloader = get_dataloader(args.data_dir, batch_size=args.batch_size, split='val')
    
    # Get input dimension and number of classes
    metadata_path = os.path.join(args.data_dir, 'metadata.json')
    with open(metadata_path, 'r') as f:
        metadata = json.load(f)
    
    num_classes = metadata['num_classes']
    input_dim = train_dataloader.dataset[0]['features'].shape[-1]
    
    print(f"Dataset size: {len(train_dataloader.dataset)} train, {len(val_dataloader.dataset)} val")
    print(f"Input dimension: {input_dim}")
    print(f"Number of classes: {num_classes}")
    
    # Create model
    model = create_model(args.model_type, input_dim, num_classes, device)
    print(f"Created model: {args.model_type}")
    
    # Define loss function and optimizer
    if args.model_type not in ['mediapipe_ml', 'mediapipe_ml_sequence']:
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(model.parameters(), lr=args.learning_rate)
    
    # Train model
    print(f"Training model: {args.model_type}")
    if args.model_type in ['mediapipe_ml', 'mediapipe_ml_sequence']:
        # Collect all training data
        X_train = []
        y_train = []
        for batch in train_dataloader:
            X_train.extend(batch['features'].numpy())
            y_train.extend(batch['label'].numpy())
        
        # Collect all validation data
        X_val = []
        y_val = []
        for batch in val_dataloader:
            X_val.extend(batch['features'].numpy())
            y_val.extend(batch['label'].numpy())
        
        # Train model
        model.fit(X_train, y_train)
        
        # Evaluate model
        train_acc = model.score(X_train, y_train)
        val_acc = model.score(X_val, y_val)
        
        print(f"Train accuracy: {train_acc:.4f}")
        print(f"Validation accuracy: {val_acc:.4f}")
        
        # Save model
        model_path = os.path.join(model_output_dir, "model_final.pkl")
        model.save(model_path)
        print(f"Model saved to {model_path}")
    else:
        # Train model
        history, best_val_acc, best_val_loss = train_model(
            model=model,
            train_dataloader=train_dataloader,
            val_dataloader=val_dataloader,
            criterion=criterion,
            optimizer=optimizer,
            device=device,
            num_epochs=args.num_epochs,
            patience=args.patience,
            checkpoint_dir=model_output_dir
        )
        
        print(f"Best validation accuracy: {best_val_acc:.4f}")
        print(f"Best validation loss: {best_val_loss:.4f}")

if __name__ == "__main__":
    main()
