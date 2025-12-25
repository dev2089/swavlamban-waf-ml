"""
Network Traffic Data Processor

This module provides functionality for processing and analyzing network traffic data
for WAF (Web Application Firewall) ML models.
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Union, Optional
from dataclasses import dataclass
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class TrafficPacket:
    """Represents a single network traffic packet."""
    timestamp: float
    src_ip: str
    dst_ip: str
    src_port: int
    dst_port: int
    protocol: str
    packet_size: int
    payload: Optional[bytes] = None
    flags: Optional[str] = None
    

class DataProcessor:
    """
    Main class for processing network traffic data.
    
    Handles data loading, cleaning, feature engineering, and normalization
    for WAF machine learning models.
    """
    
    def __init__(self, verbose: bool = True):
        """
        Initialize the DataProcessor.
        
        Args:
            verbose: Enable verbose logging output
        """
        self.verbose = verbose
        self.data = None
        self.features = None
        self.statistics = {}
        
    def load_data(self, file_path: str, file_type: str = 'csv') -> pd.DataFrame:
        """
        Load network traffic data from file.
        
        Args:
            file_path: Path to the data file
            file_type: Type of file ('csv', 'json', 'parquet')
            
        Returns:
            Loaded DataFrame
            
        Raises:
            FileNotFoundError: If file doesn't exist
            ValueError: If file type is not supported
        """
        try:
            if file_type == 'csv':
                self.data = pd.read_csv(file_path)
            elif file_type == 'json':
                self.data = pd.read_json(file_path)
            elif file_type == 'parquet':
                self.data = pd.read_parquet(file_path)
            else:
                raise ValueError(f"Unsupported file type: {file_type}")
                
            if self.verbose:
                logger.info(f"Loaded {len(self.data)} records from {file_path}")
                
            return self.data
            
        except FileNotFoundError:
            logger.error(f"File not found: {file_path}")
            raise
            
    def clean_data(self) -> pd.DataFrame:
        """
        Clean the loaded data.
        
        Handles missing values, removes duplicates, and validates data types.
        
        Returns:
            Cleaned DataFrame
        """
        if self.data is None:
            raise ValueError("No data loaded. Call load_data() first.")
            
        original_len = len(self.data)
        
        # Remove duplicates
        self.data = self.data.drop_duplicates()
        
        # Handle missing values
        missing_counts = self.data.isnull().sum()
        if missing_counts.sum() > 0:
            logger.info(f"Found {missing_counts.sum()} missing values")
            # Fill numeric columns with median
            numeric_cols = self.data.select_dtypes(include=[np.number]).columns
            self.data[numeric_cols] = self.data[numeric_cols].fillna(
                self.data[numeric_cols].median()
            )
            
        # Remove rows with critical missing values
        self.data = self.data.dropna(subset=self._get_critical_columns())
        
        records_removed = original_len - len(self.data)
        if self.verbose:
            logger.info(f"Removed {records_removed} records during cleaning")
            
        return self.data
        
    def extract_features(self) -> pd.DataFrame:
        """
        Extract and engineer features from raw traffic data.
        
        Returns:
            DataFrame with engineered features
        """
        if self.data is None:
            raise ValueError("No data loaded. Call load_data() first.")
            
        features_dict = {}
        
        # Packet-level features
        if 'packet_size' in self.data.columns:
            features_dict['packet_size'] = self.data['packet_size']
            features_dict['packet_size_squared'] = self.data['packet_size'] ** 2
            
        # Protocol features
        if 'protocol' in self.data.columns:
            protocol_dummies = pd.get_dummies(
                self.data['protocol'], 
                prefix='protocol'
            )
            features_dict.update(protocol_dummies.to_dict('series'))
            
        # Port features
        if 'dst_port' in self.data.columns:
            features_dict['dst_port'] = self.data['dst_port']
            features_dict['is_privileged_port'] = (
                self.data['dst_port'] < 1024
            ).astype(int)
            features_dict['is_well_known_port'] = self._is_well_known_port(
                self.data['dst_port']
            )
            
        # Temporal features
        if 'timestamp' in self.data.columns:
            self.data['timestamp'] = pd.to_datetime(
                self.data['timestamp'], 
                unit='s'
            )
            features_dict['hour'] = self.data['timestamp'].dt.hour
            features_dict['day_of_week'] = self.data['timestamp'].dt.dayofweek
            
        # IP-based features
        if 'src_ip' in self.data.columns:
            features_dict['src_ip_numeric'] = self._ip_to_numeric(
                self.data['src_ip']
            )
            
        if 'dst_ip' in self.data.columns:
            features_dict['dst_ip_numeric'] = self._ip_to_numeric(
                self.data['dst_ip']
            )
            
        self.features = pd.DataFrame(features_dict)
        
        if self.verbose:
            logger.info(f"Extracted {len(self.features.columns)} features")
            
        return self.features
        
    def normalize_features(self) -> pd.DataFrame:
        """
        Normalize extracted features to [0, 1] range.
        
        Returns:
            Normalized features DataFrame
        """
        if self.features is None:
            raise ValueError("No features extracted. Call extract_features() first.")
            
        normalized = self.features.copy()
        numeric_cols = normalized.select_dtypes(include=[np.number]).columns
        
        for col in numeric_cols:
            min_val = normalized[col].min()
            max_val = normalized[col].max()
            
            if max_val > min_val:
                normalized[col] = (normalized[col] - min_val) / (max_val - min_val)
            else:
                normalized[col] = 0
                
        if self.verbose:
            logger.info("Features normalized to [0, 1] range")
            
        return normalized
        
    def get_statistics(self) -> Dict:
        """
        Generate statistics about the processed data.
        
        Returns:
            Dictionary containing various statistics
        """
        if self.data is None:
            raise ValueError("No data loaded.")
            
        self.statistics = {
            'total_records': len(self.data),
            'memory_usage_mb': self.data.memory_usage(deep=True).sum() / 1024**2,
            'columns': list(self.data.columns),
            'dtypes': self.data.dtypes.to_dict(),
            'missing_values': self.data.isnull().sum().to_dict(),
        }
        
        return self.statistics
        
    def aggregate_traffic(self, 
                         group_by: List[str],
                         agg_functions: Dict[str, str]) -> pd.DataFrame:
        """
        Aggregate traffic data by specified columns.
        
        Args:
            group_by: Columns to group by
            agg_functions: Dictionary of column -> aggregation function
            
        Returns:
            Aggregated DataFrame
        """
        if self.data is None:
            raise ValueError("No data loaded.")
            
        aggregated = self.data.groupby(group_by).agg(agg_functions)
        
        if self.verbose:
            logger.info(f"Aggregated data into {len(aggregated)} groups")
            
        return aggregated
        
    def detect_anomalies(self, 
                        column: str,
                        threshold: float = 3.0) -> Tuple[np.ndarray, List[int]]:
        """
        Detect anomalies using z-score method.
        
        Args:
            column: Column to analyze
            threshold: Z-score threshold for anomaly detection
            
        Returns:
            Tuple of (z-scores, anomaly indices)
        """
        if self.data is None or column not in self.data.columns:
            raise ValueError(f"Column '{column}' not found in data")
            
        mean = self.data[column].mean()
        std = self.data[column].std()
        
        z_scores = np.abs((self.data[column] - mean) / std)
        anomaly_indices = np.where(z_scores > threshold)[0].tolist()
        
        if self.verbose:
            logger.info(f"Detected {len(anomaly_indices)} anomalies in '{column}'")
            
        return z_scores, anomaly_indices
        
    def save_processed_data(self, 
                           output_path: str,
                           file_type: str = 'csv') -> None:
        """
        Save processed data to file.
        
        Args:
            output_path: Path where to save the file
            file_type: Output file type ('csv', 'json', 'parquet')
        """
        if self.data is None:
            raise ValueError("No data to save.")
            
        try:
            if file_type == 'csv':
                self.data.to_csv(output_path, index=False)
            elif file_type == 'json':
                self.data.to_json(output_path)
            elif file_type == 'parquet':
                self.data.to_parquet(output_path, index=False)
            else:
                raise ValueError(f"Unsupported file type: {file_type}")
                
            if self.verbose:
                logger.info(f"Saved processed data to {output_path}")
                
        except Exception as e:
            logger.error(f"Error saving data: {str(e)}")
            raise
            
    # Private helper methods
    
    def _get_critical_columns(self) -> List[str]:
        """Get list of columns that are critical for analysis."""
        critical = []
        for col in ['timestamp', 'src_ip', 'dst_ip', 'protocol']:
            if col in self.data.columns:
                critical.append(col)
        return critical if critical else ['timestamp']
        
    @staticmethod
    def _ip_to_numeric(ip_series: pd.Series) -> np.ndarray:
        """Convert IP addresses to numeric values."""
        def ip_to_int(ip_str):
            try:
                parts = [int(x) for x in str(ip_str).split('.')]
                return sum(parts[i] << (8 * (3 - i)) for i in range(4))
            except (ValueError, AttributeError):
                return 0
                
        return ip_series.apply(ip_to_int).values
        
    @staticmethod
    def _is_well_known_port(port_series: pd.Series) -> np.ndarray:
        """Identify well-known ports (0-1023)."""
        return (port_series < 1024).astype(int).values


# Example usage
if __name__ == "__main__":
    processor = DataProcessor(verbose=True)
    
    # Load data
    # processor.load_data('network_traffic.csv')
    
    # Clean data
    # processor.clean_data()
    
    # Extract features
    # processor.extract_features()
    
    # Normalize features
    # normalized = processor.normalize_features()
    
    # Get statistics
    # stats = processor.get_statistics()
    # print(stats)
    
    print("DataProcessor module loaded successfully")
