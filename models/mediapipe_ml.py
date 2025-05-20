#!/usr/bin/env python
# models/mediapipe_ml.py
# MediaPipe+ML model for sign language recognition

import numpy as np
import pickle
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import joblib
import os

class MediaPipeML:
    """
    MediaPipe+ML model for sign language recognition.
    
    This model uses landmarks extracted with MediaPipe and applies machine learning
    algorithms for classification.
    """
    def __init__(self, 
                 classifier_type='svc', 
                 feature_type='all',
                 **kwargs):
        """
        Initialize the MediaPipe+ML model.
        
        Args:
            classifier_type: Type of classifier to use ('svc' or 'rf')
            feature_type: Type of features to use ('all', 'hands', 'pose', or 'hands_pose')
            **kwargs: Additional arguments for the classifier
        """
        self.classifier_type = classifier_type
        self.feature_type = feature_type
        self.classifier_args = kwargs
        self.model = None
        
        # Define the pipeline
        self._create_pipeline()
    
    def _create_pipeline(self):
        """Create the classification pipeline."""
        # Create the classifier
        if self.classifier_type == 'svc':
            classifier = SVC(
                C=10,
                kernel='rbf',
                gamma='scale',
                probability=True,
                **self.classifier_args
            )
        elif self.classifier_type == 'rf':
            classifier = RandomForestClassifier(
                n_estimators=100,
                max_depth=None,
                min_samples_split=2,
                min_samples_leaf=1,
                **self.classifier_args
            )
        else:
            raise ValueError(f"Unknown classifier type: {self.classifier_type}")
        
        # Create the pipeline
        self.model = Pipeline([
            ('scaler', StandardScaler()),
            ('classifier', classifier)
        ])
    
    def extract_features(self, landmarks, feature_type=None):
        """
        Extract features from MediaPipe landmarks.
        
        Args:
            landmarks: List of dictionaries containing landmarks for each frame
            feature_type: Type of features to use (overrides the default)
        
        Returns:
            Extracted features as a numpy array
        """
        if feature_type is None:
            feature_type = self.feature_type
        
        # Initialize lists to store features
        hand_features = []
        pose_features = []
        
        # Process each frame
        for frame in landmarks:
            frame_hand_features = []
            
            # Extract hand landmarks if they exist
            if 'left_hand' in frame and frame['left_hand'] is not None:
                left_hand = frame['left_hand'].flatten()
                frame_hand_features.append(left_hand)
            else:
                # Use zeros if hand is not detected
                frame_hand_features.append(np.zeros(21 * 3))  # 21 landmarks, 3 coordinates per landmark
            
            if 'right_hand' in frame and frame['right_hand'] is not None:
                right_hand = frame['right_hand'].flatten()
                frame_hand_features.append(right_hand)
            else:
                frame_hand_features.append(np.zeros(21 * 3))
            
            # Concatenate hand features
            hand_features.append(np.concatenate(frame_hand_features))
            
            # Extract pose landmarks if they exist
            if 'pose' in frame and frame['pose'] is not None:
                # Extract only upper body landmarks (first 25 landmarks)
                upper_body = frame['pose'][:25, :3]  # Exclude visibility
                pose_features.append(upper_body.flatten())
            else:
                pose_features.append(np.zeros(25 * 3))
        
        # Compute average features across frames
        hand_features = np.mean(hand_features, axis=0)
        pose_features = np.mean(pose_features, axis=0)
        
        # Return features based on feature_type
        if feature_type == 'hands':
            return hand_features
        elif feature_type == 'pose':
            return pose_features
        elif feature_type == 'hands_pose':
            return np.concatenate([hand_features, pose_features])
        elif feature_type == 'all':
            # Include additional features like mean, std, min, max across frames
            hand_features_std = np.std(hand_features, axis=0)
            pose_features_std = np.std(pose_features, axis=0)
            return np.concatenate([hand_features, pose_features, hand_features_std, pose_features_std])
        else:
            raise ValueError(f"Unknown feature type: {feature_type}")
    
    def preprocess_data(self, X, y):
        """
        Preprocess the data before training or inference.
        
        Args:
            X: List of landmark data
            y: List of labels
        
        Returns:
            X_features: Preprocessed features
            y: Labels
        """
        # Extract features from landmarks
        X_features = np.array([self.extract_features(landmarks) for landmarks in X])
        
        return X_features, y
    
    def fit(self, X, y):
        """
        Train the model.
        
        Args:
            X: List of landmark data
            y: List of labels
        
        Returns:
            self
        """
        # Preprocess the data
        X_features, y = self.preprocess_data(X, y)
        
        # Train the model
        self.model.fit(X_features, y)
        
        return self
    
    def predict(self, X):
        """
        Predict labels for the given data.
        
        Args:
            X: List of landmark data
        
        Returns:
            Predicted labels
        """
        # Extract features from landmarks
        X_features = np.array([self.extract_features(landmarks) for landmarks in X])
        
        # Predict labels
        return self.model.predict(X_features)
    
    def predict_proba(self, X):
        """
        Predict class probabilities for the given data.
        
        Args:
            X: List of landmark data
        
        Returns:
            Predicted class probabilities
        """
        # Extract features from landmarks
        X_features = np.array([self.extract_features(landmarks) for landmarks in X])
        
        # Predict probabilities
        return self.model.predict_proba(X_features)
    
    def score(self, X, y):
        """
        Compute the accuracy score on the given data.
        
        Args:
            X: List of landmark data
            y: List of labels
        
        Returns:
            Accuracy score
        """
        # Predict labels
        y_pred = self.predict(X)
        
        # Compute accuracy
        return accuracy_score(y, y_pred)
    
    def evaluate(self, X, y):
        """
        Evaluate the model on the given data.
        
        Args:
            X: List of landmark data
            y: List of labels
        
        Returns:
            Dictionary with evaluation metrics
        """
        # Predict labels
        y_pred = self.predict(X)
        
        # Compute evaluation metrics
        accuracy = accuracy_score(y, y_pred)
        report = classification_report(y, y_pred, output_dict=True)
        conf_matrix = confusion_matrix(y, y_pred)
        
        return {
            'accuracy': accuracy,
            'classification_report': report,
            'confusion_matrix': conf_matrix
        }
    
    def save(self, filepath):
        """
        Save the model to a file.
        
        Args:
            filepath: Path to save the model
        """
        model_data = {
            'classifier_type': self.classifier_type,
            'feature_type': self.feature_type,
            'classifier_args': self.classifier_args,
            'model': self.model
        }
        
        # Create directory if it doesn't exist
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        
        # Save the model
        with open(filepath, 'wb') as f:
            pickle.dump(model_data, f)
    
    @classmethod
    def load(cls, filepath):
        """
        Load a model from a file.
        
        Args:
            filepath: Path to the saved model
        
        Returns:
            Loaded model
        """
        with open(filepath, 'rb') as f:
            model_data = pickle.load(f)
        
        # Create a new instance
        model = cls(
            classifier_type=model_data['classifier_type'],
            feature_type=model_data['feature_type'],
            **model_data['classifier_args']
        )
        
        # Set the model
        model.model = model_data['model']
        
        return model


class MediaPipeMLSequence:
    """
    MediaPipe+ML model for sequence-based sign language recognition.
    
    This model uses landmarks extracted with MediaPipe and applies machine learning
    algorithms for sequence classification. Unlike the MediaPipeML model, this model
    considers the temporal information in the sequence.
    """
    def __init__(self, 
                 classifier_type='svc', 
                 feature_type='all',
                 sequence_pooling='mean',
                 **kwargs):
        """
        Initialize the MediaPipe+ML model for sequence classification.
        
        Args:
            classifier_type: Type of classifier to use ('svc' or 'rf')
            feature_type: Type of features to use ('all', 'hands', 'pose', or 'hands_pose')
            sequence_pooling: Method to pool sequence features ('mean', 'max', 'concat')
            **kwargs: Additional arguments for the classifier
        """
        self.classifier_type = classifier_type
        self.feature_type = feature_type
        self.sequence_pooling = sequence_pooling
        self.classifier_args = kwargs
        self.model = None
        
        # Define the pipeline
        self._create_pipeline()
    
    def _create_pipeline(self):
        """Create the classification pipeline."""
        # Create the classifier
        if self.classifier_type == 'svc':
            classifier = SVC(
                C=10,
                kernel='rbf',
                gamma='scale',
                probability=True,
                **self.classifier_args
            )
        elif self.classifier_type == 'rf':
            classifier = RandomForestClassifier(
                n_estimators=100,
                max_depth=None,
                min_samples_split=2,
                min_samples_leaf=1,
                **self.classifier_args
            )
        else:
            raise ValueError(f"Unknown classifier type: {self.classifier_type}")
        
        # Create the pipeline
        self.model = Pipeline([
            ('scaler', StandardScaler()),
            ('classifier', classifier)
        ])
    
    def extract_features_from_frame(self, frame):
        """
        Extract features from a single frame of MediaPipe landmarks.
        
        Args:
            frame: Dictionary containing landmarks for a single frame
        
        Returns:
            Extracted features as a numpy array
        """
        hand_features = []
        
        # Extract hand landmarks if they exist
        if 'left_hand' in frame and frame['left_hand'] is not None:
            left_hand = frame['left_hand'].flatten()
            hand_features.append(left_hand)
        else:
            # Use zeros if hand is not detected
            hand_features.append(np.zeros(21 * 3))  # 21 landmarks, 3 coordinates per landmark
        
        if 'right_hand' in frame and frame['right_hand'] is not None:
            right_hand = frame['right_hand'].flatten()
            hand_features.append(right_hand)
        else:
            hand_features.append(np.zeros(21 * 3))
        
        # Concatenate hand features
        hand_features = np.concatenate(hand_features)
        
        # Extract pose landmarks if they exist
        if 'pose' in frame and frame['pose'] is not None:
            # Extract only upper body landmarks (first 25 landmarks)
            upper_body = frame['pose'][:25, :3]  # Exclude visibility
            pose_features = upper_body.flatten()
        else:
            pose_features = np.zeros(25 * 3)
        
        # Return features based on feature_type
        if self.feature_type == 'hands':
            return hand_features
        elif self.feature_type == 'pose':
            return pose_features
        elif self.feature_type == 'hands_pose' or self.feature_type == 'all':
            return np.concatenate([hand_features, pose_features])
        else:
            raise ValueError(f"Unknown feature type: {self.feature_type}")
    
    def extract_features(self, landmarks):
        """
        Extract features from MediaPipe landmarks.
        
        Args:
            landmarks: List of dictionaries containing landmarks for each frame
        
        Returns:
            Extracted features as a numpy array
        """
        # Extract features from each frame
        frame_features = [self.extract_features_from_frame(frame) for frame in landmarks]
        
        # Pool features based on sequence_pooling method
        if self.sequence_pooling == 'mean':
            return np.mean(frame_features, axis=0)
        elif self.sequence_pooling == 'max':
            return np.max(frame_features, axis=0)
        elif self.sequence_pooling == 'concat':
            # Concat the first, middle, and last frames to capture the sequence
            n_frames = len(frame_features)
            if n_frames >= 3:
                first = frame_features[0]
                middle = frame_features[n_frames // 2]
                last = frame_features[-1]
                return np.concatenate([first, middle, last])
            else:
                # If there are fewer than 3 frames, use mean pooling
                return np.mean(frame_features, axis=0)
        else:
            raise ValueError(f"Unknown sequence pooling method: {self.sequence_pooling}")
    
    def preprocess_data(self, X, y):
        """
        Preprocess the data before training or inference.
        
        Args:
            X: List of landmark data
            y: List of labels
        
        Returns:
            X_features: Preprocessed features
            y: Labels
        """
        # Extract features from landmarks
        X_features = np.array([self.extract_features(landmarks) for landmarks in X])
        
        return X_features, y
    
    def fit(self, X, y):
        """
        Train the model.
        
        Args:
            X: List of landmark data
            y: List of labels
        
        Returns:
            self
        """
        # Preprocess the data
        X_features, y = self.preprocess_data(X, y)
        
        # Train the model
        self.model.fit(X_features, y)
        
        return self
    
    def predict(self, X):
        """
        Predict labels for the given data.
        
        Args:
            X: List of landmark data
        
        Returns:
            Predicted labels
        """
        # Extract features from landmarks
        X_features = np.array([self.extract_features(landmarks) for landmarks in X])
        
        # Predict labels
        return self.model.predict(X_features)
    
    def predict_proba(self, X):
        """
        Predict class probabilities for the given data.
        
        Args:
            X: List of landmark data
        
        Returns:
            Predicted class probabilities
        """
        # Extract features from landmarks
        X_features = np.array([self.extract_features(landmarks) for landmarks in X])
        
        # Predict probabilities
        return self.model.predict_proba(X_features)
    
    def score(self, X, y):
        """
        Compute the accuracy score on the given data.
        
        Args:
            X: List of landmark data
            y: List of labels
        
        Returns:
            Accuracy score
        """
        # Predict labels
        y_pred = self.predict(X)
        
        # Compute accuracy
        return accuracy_score(y, y_pred)
    
    def evaluate(self, X, y):
        """
        Evaluate the model on the given data.
        
        Args:
            X: List of landmark data
            y: List of labels
        
        Returns:
            Dictionary with evaluation metrics
        """
        # Predict labels
        y_pred = self.predict(X)
        
        # Compute evaluation metrics
        accuracy = accuracy_score(y, y_pred)
        report = classification_report(y, y_pred, output_dict=True)
        conf_matrix = confusion_matrix(y, y_pred)
        
        return {
            'accuracy': accuracy,
            'classification_report': report,
            'confusion_matrix': conf_matrix
        }
    
    def save(self, filepath):
        """
        Save the model to a file.
        
        Args:
            filepath: Path to save the model
        """
        model_data = {
            'classifier_type': self.classifier_type,
            'feature_type': self.feature_type,
            'sequence_pooling': self.sequence_pooling,
            'classifier_args': self.classifier_args,
            'model': self.model
        }
        
        # Create directory if it doesn't exist
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        
        # Save the model
        with open(filepath, 'wb') as f:
            pickle.dump(model_data, f)
    
    @classmethod
    def load(cls, filepath):
        """
        Load a model from a file.
        
        Args:
            filepath: Path to the saved model
        
        Returns:
            Loaded model
        """
        with open(filepath, 'rb') as f:
            model_data = pickle.load(f)
        
        # Create a new instance
        model = cls(
            classifier_type=model_data['classifier_type'],
            feature_type=model_data['feature_type'],
            sequence_pooling=model_data['sequence_pooling'],
            **model_data['classifier_args']
        )
        
        # Set the model
        model.model = model_data['model']
        
        return model


if __name__ == "__main__":
    # Test the models
    import argparse
    import glob
    
    parser = argparse.ArgumentParser(description="Test MediaPipe+ML models")
    parser.add_argument("--data_dir", type=str, required=True, 
                        help="Directory containing landmark data")
    parser.add_argument("--model_type", type=str, default="basic", 
                        choices=["basic", "sequence"],
                        help="Type of model to test")
    
    args = parser.parse_args()
    
    # Load landmark data
    landmark_files = glob.glob(os.path.join(args.data_dir, "landmarks", "*.pkl"))
    
    if not landmark_files:
        print(f"No landmark files found in {os.path.join(args.data_dir, 'landmarks')}")
        exit(1)
    
    print(f"Found {len(landmark_files)} landmark files")
    
    # Load a sample file to test the model
    with open(landmark_files[0], 'rb') as f:
        sample_landmarks = pickle.load(f)
    
    print(f"Sample landmarks shape: {len(sample_landmarks)} frames")
    
    # Test the model
    if args.model_type == "basic":
        model = MediaPipeML()
        features = model.extract_features(sample_landmarks)
        print(f"Extracted features shape: {features.shape}")
    else:
        model = MediaPipeMLSequence()
        features = model.extract_features(sample_landmarks)
        print(f"Extracted features shape: {features.shape}")
    
    print("Test completed successfully!")