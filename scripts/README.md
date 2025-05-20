# LinguaSign Scripts

This directory contains utility scripts for the LinguaSign project. These scripts help with common tasks such as data processing, model evaluation, and deployment.

## Available Scripts

- **evaluate_model.py**: Evaluate a trained model on a test dataset
- **convert_model.py**: Convert a model to a different format (e.g., ONNX, TorchScript)
- **benchmark.py**: Benchmark model performance

## Usage

Most scripts can be run directly from the command line with appropriate arguments. For example:

```bash
python scripts/evaluate_model.py --model_path models/checkpoints/cnn_lstm.pth --dataset datasets/processed/wlasl
```

## Adding New Scripts

When adding new scripts to this directory, please follow these guidelines:

1. Use clear, descriptive names for scripts
2. Include a docstring at the top of the script explaining its purpose
3. Add command-line arguments with helpful descriptions
4. Add the script to the list above in this README

## Script Categories

Scripts in this directory are organized into the following categories:

### Data Processing

Scripts for processing datasets, data augmentation, and feature extraction.

### Model Management

Scripts for model training, evaluation, conversion, and deployment.

### Utilities

General utility scripts for project maintenance and development.
