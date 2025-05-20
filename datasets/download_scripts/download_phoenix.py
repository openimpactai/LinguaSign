#!/usr/bin/env python
# datasets/download_scripts/download_phoenix.py
# Script to download and setup the PHOENIX-2014T dataset

import os
import argparse
import urllib.request
import subprocess
import tarfile
from pathlib import Path

def create_directory(directory):
    """Create directory if it doesn't exist."""
    os.makedirs(directory, exist_ok=True)
    print(f"Created directory: {directory}")

def download_file(url, output_path):
    """Download a file from a URL to the specified output path."""
    print(f"Downloading {url} to {output_path}...")
    urllib.request.urlretrieve(url, output_path)
    print(f"Downloaded {output_path}")

def download_phoenix(output_dir):
    """
    Download the PHOENIX-2014T dataset.
    
    Args:
        output_dir: Directory to save the dataset
    """
    # Create the output directory if it doesn't exist
    create_directory(output_dir)
    
    # URLs for PHOENIX-2014T dataset files
    # Note: These URLs are placeholders and may need to be updated with actual URLs
    # PHOENIX-2014T requires registration, so direct download links are not provided
    phoenix_urls = {
        "features": "https://www-i6.informatik.rwth-aachen.de/ftp/pub/rwth-phoenix/2014-T/phoenix-2014-T.v3.tar.gz",
        "annotations": "https://www-i6.informatik.rwth-aachen.de/ftp/pub/rwth-phoenix/2014-T/phoenix-2014-T.v3.annotations.tar.gz"
    }
    
    # Download the features
    features_path = os.path.join(output_dir, "phoenix-2014-T.v3.tar.gz")
    print("PHOENIX-2014T dataset requires registration.")
    print("Please download the dataset from:")
    print("https://www-i6.informatik.rwth-aachen.de/~koller/RWTH-PHOENIX-2014-T/")
    print(f"And save it to: {features_path}")
    
    # Ask user if they have downloaded the dataset
    user_input = input("Have you downloaded the dataset? (yes/no): ")
    if user_input.lower() != "yes":
        print("Please download the dataset and run this script again.")
        return
    
    # Check if the files exist
    if not os.path.exists(features_path):
        print(f"Error: Features file not found at {features_path}")
        return
    
    # Extract the features
    print(f"Extracting {features_path}...")
    with tarfile.open(features_path, 'r:gz') as tar:
        tar.extractall(output_dir)
    print("Extraction complete")
    
    # Download the annotations
    annotations_path = os.path.join(output_dir, "phoenix-2014-T.v3.annotations.tar.gz")
    user_input = input(f"Have you downloaded the annotations to {annotations_path}? (yes/no): ")
    if user_input.lower() != "yes":
        print("Please download the annotations and run this script again.")
        return
    
    # Check if the annotations file exists
    if not os.path.exists(annotations_path):
        print(f"Error: Annotations file not found at {annotations_path}")
        return
    
    # Extract the annotations
    print(f"Extracting {annotations_path}...")
    with tarfile.open(annotations_path, 'r:gz') as tar:
        tar.extractall(output_dir)
    print("Extraction complete")
    
    print("PHOENIX-2014T dataset setup completed!")
    print(f"Dataset is available at: {output_dir}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Download PHOENIX-2014T dataset")
    parser.add_argument("--output_dir", type=str, default="../raw/phoenix", 
                        help="Directory to save the dataset")
    
    args = parser.parse_args()
    
    # Convert relative path to absolute path
    if not os.path.isabs(args.output_dir):
        script_dir = os.path.dirname(os.path.abspath(__file__))
        output_dir = os.path.join(script_dir, args.output_dir)
    else:
        output_dir = args.output_dir
    
    download_phoenix(output_dir)
