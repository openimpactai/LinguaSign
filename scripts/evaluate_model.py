#!/usr/bin/env python
# scripts/evaluate_model.py
# Script to evaluate a trained model on a test dataset

import os
import sys
import argparse
import torch
import json
import numpy as np
from tqdm import tqdm
from sklearn.metrics import classification_report, confusion_matrix

# Add parent directory to path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from datasets.data_loader import get_dataloader
from models.cnn_lstm import CNNLSTM, LandmarkCNNLSTM
from models.mediapipe_ml import MediaPipeML
from models.transformer import TransformerModel
from models.utils import plot_confusion_matrix, print_classification_report

def load_model(model_path, model_type, num_classes, device):
    """
    Load a model from a checkpoint.
    
    Args:
        model_path: Path to the model checkpoint
        model_type: Type of model to load
        num_classes: Number of output classes
        device: Device to load the model on
    
    Returns:
        Loaded model
    """
    if model_type == 'cnn_lstm':
        if hasattr(CNNLSTM, 'load_from_checkpoint'):
            model = CNNLSTM.load_from_checkpoint(model_path)
        else:
            model = CNNLSTM(num_classes=num_classes)
            model.load_state_dict(torch.load(model_path, map_location=device))
    elif model_type == 'landmark_cnn_lstm':
        if hasattr(LandmarkCNNLSTM, 'load_from_checkpoint'):
            model = LandmarkCNNLSTM.load_from_checkpoint(model_path)
        else:
            model = LandmarkCNNLSTM(num_classes=num_classes)
            model.load_state_dict(torch.load(model_path, map_location=device))
    elif model_type == 'mediapipe_ml':
        model = MediaPipeML.load(model_path)
    elif model_type == 'transformer':
        if hasattr(TransformerModel, 'load_from_checkpoint'):
            model = TransformerModel.load_from_checkpoint(model_path)
        else:
            model = TransformerModel(num_classes=num_classes)
            model.load_state_dict(torch.load(model_path, map_location=device))
    else:
        raise ValueError(f"Unknown model type: {model_type}")
    
    # Move model to device if it's a PyTorch model
    if isinstance(model, torch.nn.Module):
        model = model.to(device)
        model.eval()
    
    return model

def evaluate_model(model, dataloader, device, model_type):
    """
    Evaluate a model on a dataset.
    
    Args:
        model: Model to evaluate
        dataloader: DataLoader for the dataset
        device: Device to evaluate on
        model_type: Type of model to evaluate
    
    Returns:
        Dictionary with evaluation results
    """
    all_labels = []
    all_predictions = []
    
    # Evaluate PyTorch models
    if isinstance(model, torch.nn.Module):
        with torch.no_grad():
            for batch in tqdm(dataloader, desc="Evaluating"):
                # Get data
                features = batch['features'].to(device)
                labels = batch['label'].to(device)
                
                # Forward pass
                outputs = model(features)
                
                # Get predictions
                _, predictions = torch.max(outputs, 1)
                
                # Store results
                all_labels.extend(labels.cpu().numpy())
                all_predictions.extend(predictions.cpu().numpy())
    # Evaluate scikit-learn models
    else:
        # Collect all data
        X = []
        y = []
        for batch in tqdm(dataloader, desc="Collecting data"):
            X.extend(batch['features'].numpy())
            y.extend(batch['label'].numpy())
        
        # Make predictions
        all_predictions = model.predict(X)
        all_labels = y
    
    # Compute evaluation metrics
    accuracy = np.mean(np.array(all_labels) == np.array(all_predictions))
    report = classification_report(all_labels, all_predictions, output_dict=True)
    conf_matrix = confusion_matrix(all_labels, all_predictions)
    
    return {
        'accuracy': accuracy,
        'classification_report': report,
        'confusion_matrix': conf_matrix,
        'predictions': all_predictions,
        'labels': all_labels
    }

def main():
    """Main function for evaluating a model."""
    parser = argparse.ArgumentParser(description="Evaluate a trained model on a test dataset")
    parser.add_argument("--model_path", type=str, required=True, 
                        help="Path to the model checkpoint")
    parser.add_argument("--model_type", type=str, default="landmark_cnn_lstm", 
                        choices=["cnn_lstm", "landmark_cnn_lstm", "mediapipe_ml", "transformer"],
                        help="Type of model to evaluate")
    parser.add_argument("--data_dir", type=str, required=True, 
                        help="Directory containing the processed dataset")
    parser.add_argument("--batch_size", type=int, default=32, 
                        help="Batch size")
    parser.add_argument("--output_dir", type=str, default=None, 
                        help="Directory to save evaluation results")
    parser.add_argument("--plot", action="store_true", 
                        help="Plot confusion matrix")
    
    args = parser.parse_args()
    
    # Determine device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Get metadata
    metadata_path = os.path.join(args.data_dir, 'metadata.json')
    with open(metadata_path, 'r') as f:
        metadata = json.load(f)
    
    num_classes = metadata['num_classes']
    
    # Load the model
    model = load_model(args.model_path, args.model_type, num_classes, device)
    print(f"Loaded model: {args.model_type}")
    
    # Get test dataloader
    test_dataloader = get_dataloader(args.data_dir, batch_size=args.batch_size, split='test')
    print(f"Test dataset size: {len(test_dataloader.dataset)}")
    
    # Evaluate the model
    results = evaluate_model(model, test_dataloader, device, args.model_type)
    
    # Print results
    print(f"Test accuracy: {results['accuracy']:.4f}")
    print("Classification report:")
    print_classification_report(results['labels'], results['predictions'])
    
    # Plot confusion matrix
    if args.plot:
        # Get class names if available
        if 'gloss_to_index' in metadata:
            gloss_to_index = metadata['gloss_to_index']
            index_to_gloss = {int(v): k for k, v in gloss_to_index.items()}
            class_names = [index_to_gloss.get(i, str(i)) for i in range(num_classes)]
        else:
            class_names = [str(i) for i in range(num_classes)]
        
        plot_confusion_matrix(results['labels'], results['predictions'], class_names)
    
    # Save results
    if args.output_dir:
        os.makedirs(args.output_dir, exist_ok=True)
        
        # Save classification report
        report_path = os.path.join(args.output_dir, 'classification_report.json')
        with open(report_path, 'w') as f:
            json.dump(results['classification_report'], f, indent=2)
        
        # Save confusion matrix
        conf_matrix_path = os.path.join(args.output_dir, 'confusion_matrix.npy')
        np.save(conf_matrix_path, results['confusion_matrix'])
        
        # Save predictions
        predictions_path = os.path.join(args.output_dir, 'predictions.npy')
        np.save(predictions_path, np.array(results['predictions']))
        
        # Save labels
        labels_path = os.path.join(args.output_dir, 'labels.npy')
        np.save(labels_path, np.array(results['labels']))
        
        print(f"Results saved to {args.output_dir}")

if __name__ == "__main__":
    main()
