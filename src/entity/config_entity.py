# Import the datetime module to work with date and time
from datetime import datetime

# Import the os module to work with file paths and directories
import os

# Import constants related to the training pipeline from our custom module
from src.constants import pipeline_constants

# (Optional) kurze Kontrolle
print(pipeline_constants.PIPELINE_NAME)
print(pipeline_constants.ARTIFACT_DIR)


class TrainingPipelineConfig:
    """
    Konfiguration für die gesamte Trainings-Pipeline.
    Erzeugt einen timestamp-basierten Artefakt-Ordner:
        artifacts/<timestamp>/
    und kennt den Modell-Ordner.
    """
    def __init__(self, timestamp: datetime = datetime.now()):
        # Timestamp formatieren wie im Beispielprojekt
        timestamp_str = timestamp.strftime("%m_%d_%Y_%H_%M_%S")

        # Name der Pipeline (aus training_pipeline.py)
        self.pipeline_name: str = pipeline_constants.PIPELINE_NAME

        # Root-Verzeichnis für Artefakte (z.B. "artifacts")
        self.artifact_name: str = pipeline_constants.ARTIFACT_DIR

        # Konkreter Artefakt-Ordner für diesen Run, z.B. artifacts/11_14_2025_15_30_12
        self.artifact_dir: str = os.path.join(self.artifact_name, timestamp_str)

        # Ordner, in dem das finale (lokale) Modell abgelegt wird
        # (z.B. "saved_models")
        #self.model_dir: str = pipeline_constants.SAVED_MODEL_DIR

        # Timestamp als String speichern
        self.timestamp: str = timestamp_str


class DataIngestionConfig:
    """
    Konfiguration für die Data-Ingestion-Stage.
    Entspricht semantisch deinem Notebook:

        DATA_ROOT   = Path("dataset")
        NORMAL_DIR  = DATA_ROOT/"normal_data"
        ANOM_DIR    = DATA_ROOT/"anomalie_data"

    plus: Pfade, wo wir ds_norm, ds_anom, ds_all und ggf. train/test speichern.
    """
    def __init__(self, training_pipeline_config: TrainingPipelineConfig):
        # Basis-Ordner für Data-Ingestion-Artefakte:
        #   artifacts/<timestamp>/data_ingestion
        self.data_ingestion_dir: str = os.path.join(
            training_pipeline_config.artifact_dir,
            pipeline_constants.DATA_INGESTION_DIR_NAME
        )

        # ---------- Eingangs-Daten (vom Board geloggte CSVs) ----------
        # Entspricht genau deinem Notebook-Setup
        self.data_root: str = pipeline_constants.DATA_ROOT
        self.normal_data_dir: str = pipeline_constants.DATA_NORMAL_DIR
        self.anomal_data_dir: str = pipeline_constants.DATA_ANOMAL_DIR

        # ---------- Ausgänge der „Notebook-Cell 1“ (ds_norm, ds_anom, ds_all) ----------
        # Wir legen sie einfach direkt unter data_ingestion_dir ab
        self.ds_norm_file_path: str = os.path.join(self.data_ingestion_dir, pipeline_constants.INGESTED_NORMAL_FILE_NAME)
        self.ds_anom_file_path: str = os.path.join(self.data_ingestion_dir, pipeline_constants.INGESTED_ANOMALY_FILE_NAME)
        self.ds_all_file_path: str  = os.path.join(self.data_ingestion_dir, pipeline_constants.RAW_COMBINED_FILE_NAME)

        self.normal_label: str = "normal" 
        self.anomal_label: str = "anomalie"


        



class DataValidationConfig:
    """
    Konfiguration für die Data-Validation-Stage.
    Arbeitet auf den Ausgaben von DataIngestion:
        - ds_norm.csv
        - ds_anom.csv
        - ds_all.csv
    """
    def __init__(self, training_pipeline_config: TrainingPipelineConfig):
        import os

        # artifacts/<ts>/data_validation
        self.data_validation_dir: str = os.path.join(
            training_pipeline_config.artifact_dir,
            pipeline_constants.DATA_VALIDATION_DIR_NAME
        )

        # wohin die validierten CSVs geschrieben werden
        self.valid_norm_file_path: str = os.path.join(
            self.data_validation_dir, pipeline_constants.VALID_NORMAL_FILE_NAME
        )
        self.valid_anom_file_path: str = os.path.join(
            self.data_validation_dir, pipeline_constants.VALID_ANOMALY_FILE_NAME
        )
        self.valid_all_file_path: str = os.path.join(
            self.data_validation_dir, pipeline_constants.VALID_COMBINED_FILE_NAME
        )


        # Pfad zur schema.yaml
        self.schema_file_path: str = pipeline_constants.SCHEMA_FILE_PATH        




class FeatureExtractionConfig:
    """
    Konfiguration für die Feature-Extraktion (Notebook Cells 2–4 + axis_scaler.h Export).

    - axis_stats_file_path: npz mit axis_mean / axis_std
    - axis_scaler_header_path: axis_scaler.h für das MCU-Projekt (emlearn)
    - train_features_file_path / test_features_file_path: npz mit X/y
    - test_size, seed: wie im Notebook
    """
    def __init__(self, training_pipeline_config: TrainingPipelineConfig):
        # Basis-Ordner für Feature-Extraction-Artefakte:
        #   artifacts/<timestamp>/feature_extraction
        self.feature_extraction_dir: str = os.path.join(
            training_pipeline_config.artifact_dir,
            pipeline_constants.FEATURE_EXTRACTION_DIR_NAME
        )

        # npz mit axis_mean / axis_std (nur NORMAL, wie im Notebook)
        self.axis_stats_file_path: str = os.path.join(
            self.feature_extraction_dir,
            pipeline_constants.AXIS_STATS_FILE_NAME
        )

        # axis_scaler.h 
        self.axis_scaler_header_path: str = os.path.join(
            self.feature_extraction_dir,
            pipeline_constants.AXIS_SCALER_FILE_NAME
        )

        # Feature-Dateien (Train/Test) als npz
        self.train_features_file_path: str = os.path.join(
            self.feature_extraction_dir,
            pipeline_constants.TRAIN_FILE_NAME
        )
        self.test_features_file_path: str = os.path.join(
            self.feature_extraction_dir,
            pipeline_constants.TEST_FILE_NAME
        )

        # Split-Parameter wie im Notebook (test_size=0.30, seed=7)
        self.test_size: float = pipeline_constants.TRAIN_TEST_SPLIT_RATIO  # entspricht test_size=0.30 im Notebook

        self.seed: int = pipeline_constants.GLOBAL_SEED



class ModelTrainerConfig:
    """
    Konfiguration für das Training + emlearn-Konvertierung.

    
    Zusätzlich:
        expected_f1: Mindest-F1, bevor Warnung ausgegeben wird
        overfitting_threshold: max. erlaubte Differenz zwischen train_F1 und test_F1
        seed: Reproduzierbarkeit
    """

    def __init__(self, training_pipeline_config: TrainingPipelineConfig):

        # Hauptverzeichnis:
        #   artifacts/<timestamp>/model_trainer
        self.model_trainer_dir: str = os.path.join(
            training_pipeline_config.artifact_dir,
            pipeline_constants.MODEL_TRAINER_DIR_NAME
        )

        # sklearn Modell speichern
        self.trained_model_file_path: str = os.path.join(
            self.model_trainer_dir,
            pipeline_constants.TRAINED_MODEL_FILE_NAME  # z.B. "model.joblib"
        )

        # MCU Codegeneration:
        #   artifacts/<timestamp>/model_trainer/codegen/

        # emlearn Header: model.h
        self.emlearn_header_file_path: str = os.path.join(
            self.model_trainer_dir,
            pipeline_constants.EMLEARN_MODEL_FILE_NAME  # z.B. "model.h"
        )

        # metadata über Modell als JSON
        self.model_info_json_path: str = os.path.join(
            self.model_trainer_dir,
            pipeline_constants.MODEL_INFO_FILE_NAME  # z.B. "model_info.json"
        )

        # Trainings-Parameter
        self.expected_f1: float = pipeline_constants.EXPECTED_F1_SCORE
        self.overfitting_threshold: float = pipeline_constants.OVERFITTING_THRESHOLD
        self.seed: int = pipeline_constants.GLOBAL_SEED