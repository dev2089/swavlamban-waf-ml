"""
ML Anomaly Detection Module for WAF
Implements Isolation Forest and Autoencoder-based anomaly detection
"""

import numpy as np
import pandas as pd
from typing import Tuple, List, Dict, Any, Optional
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, Sequential
import pickle
import joblib
from pathlib import Path


class IsolationForestDetector:
    """
    Isolation Forest-based anomaly detector for WAF logs
    Uses unsupervised learning to identify anomalous patterns
    """
    
    def __init__(self, contamination: float = 0.1, random_state: int = 42):
        """
        Initialize Isolation Forest detector
        
        Args:
            contamination: Expected proportion of anomalies (0-1)
            random_state: Random seed for reproducibility
        """
        self.model = IsolationForest(
            contamination=contamination,
            random_state=random_state,
            n_estimators=100
        )
        self.scaler = StandardScaler()
        self.is_fitted = False
    
    def fit(self, X: np.ndarray) -> None:
        """
        Fit the Isolation Forest model on normal data
        
        Args:
            X: Training data (n_samples, n_features)
        """
        X_scaled = self.scaler.fit_transform(X)
        self.model.fit(X_scaled)
        self.is_fitted = True
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Predict anomalies (-1 for anomaly, 1 for normal)
        
        Args:
            X: Data to predict (n_samples, n_features)
            
        Returns:
            Predictions array
        """
        if not self.is_fitted:
            raise ValueError("Model must be fitted before prediction")
        X_scaled = self.scaler.transform(X)
        return self.model.predict(X_scaled)
    
    def decision_function(self, X: np.ndarray) -> np.ndarray:
        """
        Get anomaly scores (-1 to 1, lower values = more anomalous)
        
        Args:
            X: Data to score
            
        Returns:
            Anomaly scores
        """
        if not self.is_fitted:
            raise ValueError("Model must be fitted before scoring")
        X_scaled = self.scaler.transform(X)
        return self.model.score_samples(X_scaled)
    
    def save(self, path: str) -> None:
        """Save model and scaler to disk"""
        model_data = {
            'model': self.model,
            'scaler': self.scaler,
            'is_fitted': self.is_fitted
        }
        joblib.dump(model_data, path)
    
    def load(self, path: str) -> None:
        """Load model and scaler from disk"""
        model_data = joblib.load(path)
        self.model = model_data['model']
        self.scaler = model_data['scaler']
        self.is_fitted = model_data['is_fitted']


class AutoencoderDetector:
    """
    Deep Autoencoder-based anomaly detector for WAF logs
    Uses reconstruction error for anomaly detection
    """
    
    def __init__(self, input_dim: int, encoding_dim: int = 8, 
                 hidden_dims: Optional[List[int]] = None):
        """
        Initialize Autoencoder detector
        
        Args:
            input_dim: Input feature dimension
            encoding_dim: Dimension of encoded representation
            hidden_dims: Dimensions of hidden layers
        """
        self.input_dim = input_dim
        self.encoding_dim = encoding_dim
        self.hidden_dims = hidden_dims or [32, 16]
        self.scaler = StandardScaler()
        self.model = None
        self.threshold = None
        self.is_fitted = False
        self._build_model()
    
    def _build_model(self) -> None:
        """Build the autoencoder architecture"""
        # Encoder
        encoder_inputs = keras.Input(shape=(self.input_dim,))
        
        # Encoder layers
        x = encoder_inputs
        for hidden_dim in self.hidden_dims:
            x = layers.Dense(hidden_dim, activation='relu')(x)
            x = layers.BatchNormalization()(x)
            x = layers.Dropout(0.2)(x)
        
        # Bottleneck
        encoded = layers.Dense(self.encoding_dim, activation='relu', name='encoding')(x)
        
        # Decoder layers (reverse of encoder)
        x = encoded
        for hidden_dim in reversed(self.hidden_dims):
            x = layers.Dense(hidden_dim, activation='relu')(x)
            x = layers.BatchNormalization()(x)
            x = layers.Dropout(0.2)(x)
        
        # Output layer
        decoded = layers.Dense(self.input_dim, activation='sigmoid')(x)
        
        # Complete autoencoder
        self.model = keras.Model(encoder_inputs, decoded)
        self.model.compile(
            optimizer=keras.optimizers.Adam(learning_rate=0.001),
            loss='mse',
            metrics=['mae']
        )
    
    def fit(self, X: np.ndarray, validation_split: float = 0.2,
            epochs: int = 50, batch_size: int = 32, verbose: int = 0) -> Dict[str, Any]:
        """
        Train the autoencoder on normal data
        
        Args:
            X: Training data
            validation_split: Validation data proportion
            epochs: Number of training epochs
            batch_size: Batch size for training
            verbose: Verbosity level
            
        Returns:
            Training history
        """
        X_scaled = self.scaler.fit_transform(X)
        
        history = self.model.fit(
            X_scaled, X_scaled,
            epochs=epochs,
            batch_size=batch_size,
            validation_split=validation_split,
            verbose=verbose,
            shuffle=True
        )
        
        # Set anomaly threshold based on training reconstruction error
        train_predictions = self.model.predict(X_scaled, verbose=0)
        train_mse = np.mean(np.square(X_scaled - train_predictions), axis=1)
        self.threshold = np.percentile(train_mse, 95)  # 95th percentile
        
        self.is_fitted = True
        return history.history
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Predict anomalies based on reconstruction error
        
        Args:
            X: Data to predict
            
        Returns:
            Predictions (-1 for anomaly, 1 for normal)
        """
        if not self.is_fitted:
            raise ValueError("Model must be fitted before prediction")
        
        X_scaled = self.scaler.transform(X)
        predictions = self.model.predict(X_scaled, verbose=0)
        mse = np.mean(np.square(X_scaled - predictions), axis=1)
        
        return np.where(mse > self.threshold, -1, 1)
    
    def reconstruction_error(self, X: np.ndarray) -> np.ndarray:
        """
        Get reconstruction error scores
        
        Args:
            X: Data to score
            
        Returns:
            Reconstruction error array
        """
        if not self.is_fitted:
            raise ValueError("Model must be fitted before scoring")
        
        X_scaled = self.scaler.transform(X)
        predictions = self.model.predict(X_scaled, verbose=0)
        return np.mean(np.square(X_scaled - predictions), axis=1)
    
    def save(self, model_path: str, metadata_path: str) -> None:
        """Save model and metadata to disk"""
        self.model.save(model_path)
        metadata = {
            'scaler': self.scaler,
            'threshold': self.threshold,
            'input_dim': self.input_dim,
            'encoding_dim': self.encoding_dim,
            'hidden_dims': self.hidden_dims,
            'is_fitted': self.is_fitted
        }
        joblib.dump(metadata, metadata_path)
    
    def load(self, model_path: str, metadata_path: str) -> None:
        """Load model and metadata from disk"""
        self.model = keras.models.load_model(model_path)
        metadata = joblib.load(metadata_path)
        self.scaler = metadata['scaler']
        self.threshold = metadata['threshold']
        self.input_dim = metadata['input_dim']
        self.encoding_dim = metadata['encoding_dim']
        self.hidden_dims = metadata['hidden_dims']
        self.is_fitted = metadata['is_fitted']


class HybridAnomalyDetector:
    """
    Combines Isolation Forest and Autoencoder for robust anomaly detection
    Uses ensemble voting for final decision
    """
    
    def __init__(self, if_contamination: float = 0.1, 
                 ae_encoding_dim: int = 8):
        """
        Initialize Hybrid detector
        
        Args:
            if_contamination: Isolation Forest contamination parameter
            ae_encoding_dim: Autoencoder encoding dimension
        """
        self.if_detector = None
        self.ae_detector = None
        self.if_contamination = if_contamination
        self.ae_encoding_dim = ae_encoding_dim
    
    def fit(self, X: np.ndarray, ae_epochs: int = 50) -> None:
        """
        Train both detectors
        
        Args:
            X: Training data
            ae_epochs: Number of epochs for autoencoder training
        """
        self.if_detector = IsolationForestDetector(
            contamination=self.if_contamination
        )
        self.if_detector.fit(X)
        
        self.ae_detector = AutoencoderDetector(
            input_dim=X.shape[1],
            encoding_dim=self.ae_encoding_dim
        )
        self.ae_detector.fit(X, epochs=ae_epochs, verbose=0)
    
    def predict(self, X: np.ndarray, voting: str = 'hard') -> np.ndarray:
        """
        Predict using both detectors
        
        Args:
            X: Data to predict
            voting: 'hard' for majority vote, 'soft' for average score
            
        Returns:
            Predictions array
        """
        if self.if_detector is None or self.ae_detector is None:
            raise ValueError("Model must be fitted before prediction")
        
        if_pred = self.if_detector.predict(X)
        ae_pred = self.ae_detector.predict(X)
        
        if voting == 'hard':
            # Majority voting
            ensemble_pred = np.sign((if_pred + ae_pred) / 2)
            return np.where(ensemble_pred < 0, -1, 1)
        else:
            # Soft voting using anomaly scores
            if_scores = self.if_detector.decision_function(X)
            ae_scores = -self.ae_detector.reconstruction_error(X)  # Negate for consistency
            combined_scores = (if_scores + ae_scores) / 2
            return combined_scores
    
    def get_anomaly_scores(self, X: np.ndarray) -> Dict[str, np.ndarray]:
        """
        Get individual anomaly scores from both detectors
        
        Args:
            X: Data to score
            
        Returns:
            Dictionary with scores from each detector
        """
        if self.if_detector is None or self.ae_detector is None:
            raise ValueError("Model must be fitted before scoring")
        
        return {
            'isolation_forest': self.if_detector.decision_function(X),
            'autoencoder': -self.ae_detector.reconstruction_error(X),
            'reconstruction_error': self.ae_detector.reconstruction_error(X)
        }
    
    def save(self, directory: str) -> None:
        """Save both models to directory"""
        Path(directory).mkdir(parents=True, exist_ok=True)
        self.if_detector.save(f"{directory}/isolation_forest.pkl")
        self.ae_detector.save(
            f"{directory}/autoencoder_model.h5",
            f"{directory}/autoencoder_metadata.pkl"
        )
    
    def load(self, directory: str) -> None:
        """Load both models from directory"""
        self.if_detector = IsolationForestDetector()
        self.if_detector.load(f"{directory}/isolation_forest.pkl")
        
        self.ae_detector = AutoencoderDetector(input_dim=1)  # Placeholder
        self.ae_detector.load(
            f"{directory}/autoencoder_model.h5",
            f"{directory}/autoencoder_metadata.pkl"
        )


def create_sample_data(n_samples: int = 1000, 
                       n_features: int = 10,
                       anomaly_ratio: float = 0.05) -> Tuple[np.ndarray, np.ndarray]:
    """
    Create sample WAF log data for testing
    
    Args:
        n_samples: Number of samples
        n_features: Number of features
        anomaly_ratio: Proportion of anomalies
        
    Returns:
        Data and labels (0 for normal, 1 for anomaly)
    """
    # Normal data
    normal_samples = int(n_samples * (1 - anomaly_ratio))
    X_normal = np.random.randn(normal_samples, n_features) * 0.5 + 5
    
    # Anomalous data
    anomaly_samples = int(n_samples * anomaly_ratio)
    X_anomaly = np.random.randn(anomaly_samples, n_features) * 2 + 15
    
    X = np.vstack([X_normal, X_anomaly])
    y = np.hstack([np.zeros(normal_samples), np.ones(anomaly_samples)])
    
    # Shuffle
    shuffle_idx = np.random.permutation(len(X))
    return X[shuffle_idx], y[shuffle_idx]


if __name__ == "__main__":
    # Example usage
    print("ML Anomaly Detection Module for WAF")
    print("=" * 50)
    
    # Create sample data
    print("\nGenerating sample data...")
    X, y_true = create_sample_data(n_samples=500, n_features=15, anomaly_ratio=0.1)
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y_true, test_size=0.3, random_state=42
    )
    
    print(f"Training samples: {len(X_train)}")
    print(f"Test samples: {len(X_test)}")
    
    # Test Isolation Forest
    print("\n" + "=" * 50)
    print("Testing Isolation Forest Detector")
    print("=" * 50)
    if_detector = IsolationForestDetector(contamination=0.1)
    if_detector.fit(X_train)
    if_pred = if_detector.predict(X_test)
    print(f"Detected anomalies: {np.sum(if_pred == -1)} / {len(X_test)}")
    
    # Test Autoencoder
    print("\n" + "=" * 50)
    print("Testing Autoencoder Detector")
    print("=" * 50)
    ae_detector = AutoencoderDetector(input_dim=X.shape[1], encoding_dim=8)
    history = ae_detector.fit(X_train, epochs=30, verbose=0)
    ae_pred = ae_detector.predict(X_test)
    print(f"Detected anomalies: {np.sum(ae_pred == -1)} / {len(X_test)}")
    
    # Test Hybrid Detector
    print("\n" + "=" * 50)
    print("Testing Hybrid Detector")
    print("=" * 50)
    hybrid_detector = HybridAnomalyDetector(if_contamination=0.1, ae_encoding_dim=8)
    hybrid_detector.fit(X_train, ae_epochs=30)
    hybrid_pred = hybrid_detector.predict(X_test, voting='hard')
    print(f"Detected anomalies: {np.sum(hybrid_pred == -1)} / {len(X_test)}")
    
    # Get anomaly scores
    scores = hybrid_detector.get_anomaly_scores(X_test[:10])
    print("\nSample anomaly scores (first 10 samples):")
    for detector_name, detector_scores in scores.items():
        print(f"{detector_name}: {detector_scores[:5]}")
