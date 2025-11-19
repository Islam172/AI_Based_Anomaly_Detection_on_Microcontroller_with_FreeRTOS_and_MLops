# Import the dataclass decorator to simplify class creation for data storage
from dataclasses import dataclass 
from pathlib import Path
from typing import Optional

# Define a data class to store file paths generated during data ingestion
@dataclass
class DataIngestionArtifact:
    normal_file_path: Path
    anomal_file_path: Path
    combined_file_path: Path
    

@dataclass
class DataValidationArtifact:
    
    
    valid_norm_file_path: Path
    valid_anom_file_path: Path
    valid_all_file_path: Path
    

@dataclass
class FeatureExtractionArtifact:
    
    axis_stats_file_path: Path           # npz mit axis_mean / axis_std
    axis_scaler_header_path: Path       # axis_scaler.h für MCU
    train_features_file_path: Path      # npz mit X_train, y_train
    test_features_file_path: Path       # npz mit X_test, y_test
    n_train_windows: int
    n_test_windows: int



@dataclass
class ClassificationMetricArtifact:
    f1_score: float
    precision_score: float
    recall_score: float
    roc_auc: Optional[float] = None

@dataclass
class ModelTrainerArtifact:
    trained_model_file_path: Path
    emlearn_header_file_path: Path
    model_info_json_path: Path
    train_metric: ClassificationMetricArtifact
    test_metric: ClassificationMetricArtifact
    best_params: dict
