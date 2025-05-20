# LinguaSign Datasets

This directory contains scripts and configurations for sign language datasets used in the LinguaSign project. The actual dataset files are not stored in this repository due to size constraints.

## Supported Datasets

### 1. WLASL (Word-Level American Sign Language)
- **Description**: Contains 14,289 video segments of 2,000 common ASL signs
- **Source**: [WLASL GitHub Repository](https://github.com/dxli94/WLASL)
- **Paper**: [Word-level Deep Sign Language Recognition from Video](https://arxiv.org/abs/1910.11006)
- **Download**: Use `download_scripts/download_wlasl.py` to download and setup this dataset

### 2. PHOENIX-2014T (German Sign Language)
- **Description**: Contains 7,096 training videos from weather forecasts
- **Source**: [PHOENIX-2014T Dataset](https://www-i6.informatik.rwth-aachen.de/~koller/RWTH-PHOENIX-2014-T/)
- **Paper**: [Neural Sign Language Translation](http://www-i6.informatik.rwth-aachen.de/publications/download/1106/CamgozHadfield-MenezesKollerNeyBowden--NeuralSignLanguageTranslation--2018.pdf)
- **Download**: Use `download_scripts/download_phoenix.py` to download and setup this dataset

### 3. How2Sign (American Sign Language)
- **Description**: Contains 31,128 training videos linked to instructional content
- **Source**: [How2Sign Dataset](https://how2sign.github.io/)
- **Paper**: [How2Sign: A Large-scale Multimodal Dataset for Continuous American Sign Language](https://arxiv.org/abs/2008.08143)
- **Download**: Use `download_scripts/download_how2sign.py` to download and setup this dataset

## Dataset Directory Structure

After running the download scripts, your dataset directory should have the following structure:

```
datasets/
├── raw/                   # Raw dataset files 
│   ├── wlasl/             # WLASL dataset
│   ├── phoenix/           # PHOENIX-2014T dataset
│   └── how2sign/          # How2Sign dataset
│
├── processed/             # Preprocessed datasets ready for training
│   ├── wlasl/
│   ├── phoenix/
│   └── how2sign/
│
└── features/              # Extracted features from raw datasets
    ├── wlasl/
    ├── phoenix/
    └── how2sign/
```

## Usage Instructions

1. Install required dependencies:
   ```
   pip install -r requirements.txt
   ```

2. Download a dataset (e.g., WLASL):
   ```
   python download_scripts/download_wlasl.py
   ```

3. Preprocess the dataset:
   ```
   python preprocessing/preprocess_wlasl.py
   ```

4. The processed dataset is now ready to be used for training your models.

## Adding New Datasets

To add support for a new dataset:
1. Create a download script in the `download_scripts` directory
2. Create a preprocessing script in the `preprocessing` directory
3. Update this README with information about the new dataset
4. Update the dataset loader in the main codebase to support the new dataset
