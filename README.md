# LinguaSign: AI-powered Sign Language Translation and Learning Assistant

LinguaSign is an open-source initiative dedicated to breaking down communication barriers between the deaf community and the hearing world through artificial intelligence. This project aims to create a comprehensive system for sign language translation and learning assistance.

## Features

- **Sign-to-Text Translation**: Convert sign language gestures from video to written text
- **Text-to-Sign Visualization**: Generate visual representations of signs from text input
- **Interactive Learning**: Provide feedback and guidance for learning sign language
- **Multi-model Support**: Implements various state-of-the-art AI architectures
- **Multiple Dataset Integration**: Works with various sign language datasets

## Project Structure

```
LinguaSign/
├── api/                  # Backend API endpoints
├── datasets/             # Dataset handling and preprocessing
│   ├── download_scripts/ # Scripts to download datasets
│   └── preprocessing/    # Scripts for data preprocessing
├── docs/                 # Documentation
├── frontend/             # User interface
├── models/               # Model definitions and training scripts
│   ├── cnn_lstm.py       # CNN+LSTM hybrid model
│   ├── mediapipe_ml.py   # MediaPipe+ML model
│   ├── transformer.py    # Transformer-based model
│   └── utils.py          # Utility functions for models
├── notebooks/            # Exploratory analysis and experiments
├── scripts/              # Utility scripts
├── tests/                # Test cases
├── .gitignore           
├── LICENSE               # MIT License
├── README.md             # This file
└── requirements.txt      # Python dependencies
```

## Supported Datasets

LinguaSign is designed to work with multiple sign language datasets:

- **WLASL** (Word-Level American Sign Language): 14,289 videos of 2,000 common ASL signs
- **PHOENIX-2014T** (German Sign Language): 7,096 videos from weather forecasts
- **How2Sign** (American Sign Language): 31,128 videos of instructional content

## Model Architectures

LinguaSign implements three main model architectures:

1. **CNN+LSTM Hybrid**: Combines CNN for spatial feature extraction with LSTM for temporal dependencies
2. **MediaPipe+ML**: Uses MediaPipe for landmark extraction and traditional machine learning for classification
3. **Transformer-based**: Leverages transformer architecture for sequence modeling

## Getting Started

### Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/openimpactai/LinguaSign.git
   cd LinguaSign
   ```

2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

3. Download and preprocess a dataset:
   ```bash
   python datasets/download_scripts/download_wlasl.py
   python datasets/preprocessing/preprocess_wlasl.py
   ```

### Training a Model

```bash
python models/training.py --model_type landmark_cnn_lstm --data_dir datasets/processed/wlasl --output_dir checkpoints
```

Available model types:
- `cnn_lstm`: CNN+LSTM model for raw video input
- `landmark_cnn_lstm`: CNN+LSTM model for landmark input
- `mediapipe_ml`: MediaPipe+ML model
- `mediapipe_ml_sequence`: MediaPipe+ML model with sequence information
- `transformer`: Transformer model

### Using a Pretrained Model

```python
from models.cnn_lstm import LandmarkCNNLSTM

# Load pretrained model
model = LandmarkCNNLSTM.load_from_checkpoint('models/pretrained/landmark_cnn_lstm.pth')

# Use the model for inference
model.eval()
outputs = model(inputs)
```

## Contributing

We welcome contributions from the community! Please check out our [contribution guidelines](docs/CONTRIBUTING.md) for details on how to get involved.

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Acknowledgements

- We thank the creators of the datasets used in this project for their valuable contributions to sign language research.
- Special thanks to the deaf and hard-of-hearing community for their guidance and feedback.

## Contact

For questions or feedback, please open an issue on GitHub or contact the maintainers at [contact@openimpactai.org](mailto:contact@openimpactai.org).
