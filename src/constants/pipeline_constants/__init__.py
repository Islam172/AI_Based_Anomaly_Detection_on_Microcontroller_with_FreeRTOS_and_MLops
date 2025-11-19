import os
import sys
import numpy as np
import pandas as pd

"""
Defining common constant variables for the MCU vibration anomaly project
"""

# Zielspalte in unseren Daten (aus dem Notebook: "normal" / "anomalie")
TARGET_COLUMN: str = "label"

# Name der gesamten Pipeline
PIPELINE_NAME: str = "McuVibrationAnomaly"

# Wo alle Artefakte der Pipeline landen (Logs, Zwischendaten, Modelle, …)
ARTIFACT_DIR: str = "artifacts"

# --------------------------------------------------------------------
# Rohdaten / Dataset-Struktur (entspricht deinem Notebook: Cell 0)
# --------------------------------------------------------------------

# Root-Ordner für die geloggten CSVs vom Board
DATA_ROOT: str = os.path.join("dataset")

# Unterordner für Normal- und Anomalie-Daten
DATA_NORMAL_DIR: str = os.path.join(DATA_ROOT, "normal_data")
DATA_ANOMAL_DIR: str = os.path.join(DATA_ROOT, "anomalie_data")








# --------------------------------------------------------------------
# Signal-/Window-Parameter (müssen zum MCU passen)
# --------------------------------------------------------------------
FS_HZ: int = 200    # Sampling freq
WIN: int = 128      # Fensterlänge
HOP: int = 64       # Hop size
N_CH: int = 3       # Ax, Ay, Az

FEATS_PER_CH: int = 2  # mean + std
FEAT_DIM: int = N_CH * FEATS_PER_CH

GLOBAL_SEED: int = 7   # entspricht np.random.seed(7)

"""
Data Ingestion related constants (angepasst an dataset-Ordner, kein Mongo, kein Feature-Store)
"""
DATA_INGESTION_DIR_NAME: str = "data_ingestion"

# Wohin DataIngestion ihre CSV-Artefakte schreiben kann (z.B. ds_norm, ds_anom, ds_all)
DATA_INGESTION_OUTPUT_DIR: str = os.path.join(ARTIFACT_DIR, DATA_INGESTION_DIR_NAME)

# Ein „kombiniertes“ Rohdaten-File (falls du alles zusammen speichern willst)
# Entspricht inhaltlich dem, was ds_all wäre.
RAW_COMBINED_FILE_NAME: str = "ds_all.csv"
INGESTED_NORMAL_FILE_NAME: str = "ds_norm.csv"
INGESTED_ANOMALY_FILE_NAME: str = "ds_anom.csv"



"""
Data validation related constants
"""
DATA_VALIDATION_DIR_NAME: str = "data_validation"
SCHEMA_FILE_PATH: str = os.path.join("data_schema", "schema.yaml")
VALID_COMBINED_FILE_NAME: str = "valid_ds_all.csv"
VALID_NORMAL_FILE_NAME: str = "valid_ds_norm.csv"
VALID_ANOMALY_FILE_NAME: str = "valid_ds_anom.csv"






"""
feature extraction related constants

"""
FEATURE_EXTRACTION_DIR_NAME: str = "feature_extraction"
# Falls du später einen klassischen Train/Test-Split auf CSV-Ebene nutzen willst:
TRAIN_TEST_SPLIT_RATIO: float = 0.30  # entspricht test_size=0.30 im Notebook

AXIS_STATS_FILE_NAME: str = "axis_stats.npz"
AXIS_SCALER_FILE_NAME: str = "axis_scaler.h"

TRAIN_FILE_NAME: str = "train_features.npz"
TEST_FILE_NAME: str = "test_features.npz"





"""
Model training related constants
"""
MODEL_TRAINING_DIR_NAME: str = "model_training"
# --------------------------------------------------------------------
# Modell-Speicherort (Notebook speichert rf_model.joblib)
# --------------------------------------------------------------------


# --------------------------------------------------------------------
# emlearn / RF-Codegen-Ausgabe (Notebook: OUT_DIR, CODEGEN_DIR)
# --------------------------------------------------------------------



MODEL_TRAINER_DIR_NAME = "model_trainer"
TRAINED_MODEL_FILE_NAME = "model.joblib"


EMLEARN_MODEL_FILE_NAME = "model.h"
MODEL_INFO_FILE_NAME = "model_info.json"

EXPECTED_F1_SCORE = 0.80
OVERFITTING_THRESHOLD = 0.10
