# LinguaSign Models

This directory contains model definitions and training scripts for the LinguaSign project. Three main model architectures are implemented:

1. **CNN+LSTM Hybrid** - A model that uses CNN to extract spatial features from frames and LSTM to capture temporal dependencies.
2. **MediaPipe+ML** - A model that uses MediaPipe for landmark extraction and traditional machine learning for classification.
3. **Transformer-based** - A model that uses transformers for sequence modeling.

## Model Architecture Overview

### 1. CNN+LSTM Hybrid Model

The CNN+LSTM model combines a convolutional neural network (CNN) for feature extraction with a long short-term memory (LSTM) network for sequence modeling. This architecture is effective for sign language recognition because:

- CNN extracts spatial features from each frame
- LSTM captures temporal dependencies across frames
- End-to-end learning aligns spatial and temporal features

```
Input Video → CNN → Frame Features → LSTM → Sequence Features → Dense → Output
```

### 2. MediaPipe+ML Model

The MediaPipe+ML model uses Google's MediaPipe framework to extract hand, face, and pose landmarks from each frame, and then applies machine learning algorithms to classify the sign. This architecture is effective because:

- MediaPipe provides robust hand, face, and pose tracking
- Feature engineering is simplified with pre-defined landmarks
- Traditional ML algorithms can be fast and efficient for classification

```
Input Video → MediaPipe → Landmarks → Feature Engineering → ML Model → Output
```

### 3. Transformer-based Model

The Transformer-based model uses self-attention mechanisms to model relationships between all frames in a sequence. This architecture is effective for sign language recognition because:

- Self-attention captures long-range dependencies
- Parallel processing improves training efficiency
- State-of-the-art performance on sequence modeling tasks

```
Input Video → Frame Embeddings → Transformer Encoder → Sequence Features → Dense → Output
```

## Model Implementation Files

- `cnn_lstm.py` - Implementation of the CNN+LSTM hybrid model
- `mediapipe_ml.py` - Implementation of the MediaPipe+ML model
- `transformer.py` - Implementation of the Transformer-based model
- `training.py` - Training script for all models
- `utils.py` - Utility functions for model training and evaluation

## Usage

To train a model, use the `training.py` script:

```bash
python training.py --model_type cnn_lstm --data_dir ../datasets/processed/wlasl --output_dir ./checkpoints/cnn_lstm
```

Available model types:
- `cnn_lstm` - CNN+LSTM hybrid model
- `mediapipe_ml` - MediaPipe+ML model
- `transformer` - Transformer-based model

## Adding New Models

To add a new model architecture:

1. Create a new file with your model implementation (e.g., `new_model.py`)
2. Implement the model following the same interface as the existing models
3. Update `training.py` to support the new model type
4. Update this README with information about the new model

## Pretrained Models

Pretrained models are available in the `pretrained` directory. To use a pretrained model:

```python
from models.cnn_lstm import CNNLSTM

# Load pretrained model
model = CNNLSTM.load_from_checkpoint('models/pretrained/cnn_lstm.pth')

# Use the model for inference
model.eval()
outputs = model(inputs)
```
