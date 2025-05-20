#!/usr/bin/env python
# models/utils.py
# Utility functions for model training and evaluation

import torch
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, classification_report
import seaborn as sns
import os
import cv2
import mediapipe as mp

# Initialize MediaPipe solutions
mp_holistic = mp.solutions.holistic
mp_drawing = mp.solutions.drawing_utils

def plot_training_history(history, output_dir=None):
    """
    Plot training history.
    
    Args:
        history: Dictionary with training history
        output_dir: Directory to save plots (optional)
    """
    # Create a figure with two subplots
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
    
    # Plot loss
    ax1.plot(history['train_loss'], label='train')
    ax1.plot(history['val_loss'], label='val')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Loss')
    ax1.set_title('Loss')
    ax1.legend()
    
    # Plot accuracy
    ax2.plot(history['train_acc'], label='train')
    ax2.plot(history['val_acc'], label='val')
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Accuracy')
    ax2.set_title('Accuracy')
    ax2.legend()
    
    # Adjust layout
    plt.tight_layout()
    
    # Save plot if output_dir is provided
    if output_dir is not None:
        os.makedirs(output_dir, exist_ok=True)
        plt.savefig(os.path.join(output_dir, 'training_history.png'))
    
    # Show plot
    plt.show()

def plot_confusion_matrix(y_true, y_pred, class_names=None, output_dir=None):
    """
    Plot confusion matrix.
    
    Args:
        y_true: True labels
        y_pred: Predicted labels
        class_names: List of class names (optional)
        output_dir: Directory to save plot (optional)
    """
    # Compute confusion matrix
    cm = confusion_matrix(y_true, y_pred)
    
    # Create figure
    plt.figure(figsize=(10, 8))
    
    # Plot confusion matrix
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=False,
                xticklabels=class_names, yticklabels=class_names)
    
    # Set labels
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.title('Confusion Matrix')
    
    # Adjust layout
    plt.tight_layout()
    
    # Save plot if output_dir is provided
    if output_dir is not None:
        os.makedirs(output_dir, exist_ok=True)
        plt.savefig(os.path.join(output_dir, 'confusion_matrix.png'))
    
    # Show plot
    plt.show()

def print_classification_report(y_true, y_pred, class_names=None, output_dir=None):
    """
    Print classification report.
    
    Args:
        y_true: True labels
        y_pred: Predicted labels
        class_names: List of class names (optional)
        output_dir: Directory to save report (optional)
    """
    # Compute classification report
    report = classification_report(y_true, y_pred, target_names=class_names)
    
    # Print report
    print(report)
    
    # Save report if output_dir is provided
    if output_dir is not None:
        os.makedirs(output_dir, exist_ok=True)
        with open(os.path.join(output_dir, 'classification_report.txt'), 'w') as f:
            f.write(report)

def extract_landmarks_from_video(video_path, min_detection_confidence=0.5, display=False):
    """
    Extract landmarks from a video using MediaPipe Holistic.
    
    Args:
        video_path: Path to the video file
        min_detection_confidence: Minimum confidence value for detection to be considered successful
        display: Whether to display the video with landmarks
    
    Returns:
        List of dictionaries containing landmarks for each frame
    """
    # Open the video file
    cap = cv2.VideoCapture(video_path)
    
    # Check if the video was opened successfully
    if not cap.isOpened():
        print(f"Error: Could not open video {video_path}")
        return None
    
    # Get video properties
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    
    # Initialize MediaPipe Holistic
    with mp_holistic.Holistic(
        min_detection_confidence=min_detection_confidence,
        min_tracking_confidence=0.5) as holistic:
        
        # Initialize list to store landmarks
        frame_landmarks = []
        
        # Initialize video writer if display is True
        if display:
            fourcc = cv2.VideoWriter_fourcc(*'XVID')
            output_path = os.path.splitext(video_path)[0] + '_landmarks.avi'
            out = cv2.VideoWriter(output_path, fourcc, fps, (frame_width, frame_height))
        
        # Process each frame in the video
        while cap.isOpened():
            success, image = cap.read()
            if not success:
                break
            
            # Convert the BGR image to RGB
            image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            
            # Process the image and extract landmarks
            results = holistic.process(image_rgb)
            
            # Store the landmarks
            frame_data = {
                'pose': None,
                'face': None,
                'left_hand': None,
                'right_hand': None
            }
            
            # Convert landmarks to numpy arrays if they exist
            if results.pose_landmarks:
                pose = np.array([[lm.x, lm.y, lm.z, lm.visibility] for lm in results.pose_landmarks.landmark])
                frame_data['pose'] = pose
            
            if results.face_landmarks:
                face = np.array([[lm.x, lm.y, lm.z] for lm in results.face_landmarks.landmark])
                frame_data['face'] = face
            
            if results.left_hand_landmarks:
                left_hand = np.array([[lm.x, lm.y, lm.z] for lm in results.left_hand_landmarks.landmark])
                frame_data['left_hand'] = left_hand
            
            if results.right_hand_landmarks:
                right_hand = np.array([[lm.x, lm.y, lm.z] for lm in results.right_hand_landmarks.landmark])
                frame_data['right_hand'] = right_hand
            
            frame_landmarks.append(frame_data)
            
            # Draw landmarks on the image if display is True
            if display:
                # Convert the RGB image back to BGR
                image = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2BGR)
                
                # Draw landmarks
                if results.pose_landmarks:
                    mp_drawing.draw_landmarks(
                        image, results.pose_landmarks, mp_holistic.POSE_CONNECTIONS)
                
                if results.face_landmarks:
                    mp_drawing.draw_landmarks(
                        image, results.face_landmarks, mp_holistic.FACEMESH_TESSELATION)
                
                if results.left_hand_landmarks:
                    mp_drawing.draw_landmarks(
                        image, results.left_hand_landmarks, mp_holistic.HAND_CONNECTIONS)
                
                if results.right_hand_landmarks:
                    mp_drawing.draw_landmarks(
                        image, results.right_hand_landmarks, mp_holistic.HAND_CONNECTIONS)
                
                # Write frame to output video
                out.write(image)
                
                # Display the image
                cv2.imshow('MediaPipe Holistic', image)
                if cv2.waitKey(5) & 0xFF == 27:  # ESC key
                    break
        
        # Release the video capture and writer
        cap.release()
        if display:
            out.release()
            cv2.destroyAllWindows()
            print(f"Landmarks video saved to {output_path}")
    
    return frame_landmarks

def predict_from_video(model, video_path, device, min_detection_confidence=0.5, display=False):
    """
    Predict sign from a video.
    
    Args:
        model: Trained model
        video_path: Path to the video file
        device: Device to run the model on
        min_detection_confidence: Minimum confidence value for detection to be considered successful
        display: Whether to display the video with landmarks and prediction
    
    Returns:
        Predicted class
    """
    # Extract landmarks from the video
    landmarks = extract_landmarks_from_video(video_path, min_detection_confidence, display)
    
    if landmarks is None:
        return None
    
    # Convert landmarks to features
    if isinstance(model, (torch.nn.Module)):
        # PyTorch model
        model.eval()
        with torch.no_grad():
            # Prepare input for model
            features = np.array([landmarks])
            features = torch.from_numpy(features).float().to(device)
            
            # Get prediction
            outputs = model(features)
            _, predicted = torch.max(outputs, 1)
            
            return predicted.item()
    else:
        # Scikit-learn model
        return model.predict([landmarks])[0]

def convert_class_to_text(predicted_class, class_names):
    """
    Convert predicted class to text.
    
    Args:
        predicted_class: Predicted class index
        class_names: List of class names
    
    Returns:
        Predicted class name
    """
    if predicted_class < 0 or predicted_class >= len(class_names):
        return "Unknown"
    
    return class_names[predicted_class]

if __name__ == "__main__":
    # Test extract_landmarks_from_video function
    import argparse
    
    parser = argparse.ArgumentParser(description="Extract landmarks from a video")
    parser.add_argument("--video_path", type=str, required=True, 
                        help="Path to the video file")
    parser.add_argument("--display", action="store_true", 
                        help="Display the video with landmarks")
    
    args = parser.parse_args()
    
    # Extract landmarks from the video
    landmarks = extract_landmarks_from_video(args.video_path, display=args.display)
    
    # Print number of frames
    print(f"Extracted landmarks from {len(landmarks)} frames")
    
    # Print landmark shapes
    for key in ['pose', 'face', 'left_hand', 'right_hand']:
        if landmarks[0][key] is not None:
            print(f"{key} shape: {landmarks[0][key].shape}")
        else:
            print(f"{key} shape: None")
