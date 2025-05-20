#!/usr/bin/env python
# datasets/download_scripts/download_wlasl.py
# Script to download and setup the WLASL dataset

import os
import json
import argparse
import subprocess
from pathlib import Path
import urllib.request
import zipfile

def create_directory(directory):
    """Create directory if it doesn't exist."""
    os.makedirs(directory, exist_ok=True)
    print(f"Created directory: {directory}")

def download_file(url, output_path):
    """Download a file from a URL to the specified output path."""
    print(f"Downloading {url} to {output_path}...")
    urllib.request.urlretrieve(url, output_path)
    print(f"Downloaded {output_path}")

def download_wlasl(output_dir, version="v0.3"):
    """
    Download the WLASL dataset.
    
    Args:
        output_dir: Directory to save the dataset
        version: Version of the WLASL dataset to download
    """
    # Create the output directory if it doesn't exist
    create_directory(output_dir)
    
    # URLs for WLASL dataset files
    wlasl_urls = {
        "repo_zip": "https://github.com/dxli94/WLASL/archive/refs/heads/master.zip",
        "json_file": f"https://github.com/dxli94/WLASL/raw/master/WLASL_{version}.json"
    }
    
    # Download the repository (contains metadata and scripts)
    repo_zip_path = os.path.join(output_dir, "wlasl_repo.zip")
    download_file(wlasl_urls["repo_zip"], repo_zip_path)
    
    # Extract the repository
    print(f"Extracting {repo_zip_path}...")
    with zipfile.ZipFile(repo_zip_path, 'r') as zip_ref:
        zip_ref.extractall(output_dir)
    print("Extraction complete")
    
    # Download the JSON file (contains video metadata)
    json_path = os.path.join(output_dir, f"WLASL_{version}.json")
    download_file(wlasl_urls["json_file"], json_path)
    
    # Setup directories for videos
    videos_dir = os.path.join(output_dir, "videos")
    create_directory(videos_dir)
    
    # Parse the JSON file to get video URLs
    with open(json_path, 'r') as f:
        data = json.load(f)
    
    # Download script path from the repository
    download_script_path = os.path.join(output_dir, "WLASL-master", "start_kit", "video_downloader.py")
    
    # Check if the video downloader script exists
    if not os.path.exists(download_script_path):
        print(f"Error: Video downloader script not found at {download_script_path}")
        return
    
    # Run the video downloader script to download the videos
    print("Downloading videos using the WLASL video downloader script...")
    subprocess.run(["python", download_script_path, json_path, videos_dir])
    
    print("WLASL dataset download completed!")
    print(f"Dataset is available at: {output_dir}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Download WLASL dataset")
    parser.add_argument("--output_dir", type=str, default="../raw/wlasl", 
                        help="Directory to save the dataset")
    parser.add_argument("--version", type=str, default="v0.3", 
                        help="Version of the WLASL dataset to download")
    
    args = parser.parse_args()
    
    # Convert relative path to absolute path
    if not os.path.isabs(args.output_dir):
        script_dir = os.path.dirname(os.path.abspath(__file__))
        output_dir = os.path.join(script_dir, args.output_dir)
    else:
        output_dir = args.output_dir
    
    download_wlasl(output_dir, args.version)
