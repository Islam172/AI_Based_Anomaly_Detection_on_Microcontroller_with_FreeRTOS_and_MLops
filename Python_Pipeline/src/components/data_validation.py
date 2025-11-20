# src/pipeline/data_validation.py

from __future__ import annotations

import os
import sys
from pathlib import Path

import pandas as pd


from src.entity.artifact_entity import DataIngestionArtifact, DataValidationArtifact
from src.entity.config_entity import DataValidationConfig
from src.constants import pipeline_constants
from src.utils.utils import read_yaml_file
from src.logging.logger import logging
from src.exception.exception import Exception


class DataValidation:
    """
    DataValidation für dein Vibrations-Projekt.

    Aufgaben:
    1. Schema aus schema.yaml lesen (Spalten, numerische Spalten, etc.).
    2. Spaltenanzahl / -namen gegen Schema prüfen.
    3. KS-Test (numerische Spalten) zwischen normal und anomal als „Drift-Report“.
    4. Validierte CSVs + YAML-Report abspeichern.
    """

    def __init__(
        self,
        data_ingestion_artifact: DataIngestionArtifact,
        data_validation_config: DataValidationConfig,
    ):
        try:
            self.data_ingestion_artifact = data_ingestion_artifact
            self.config = data_validation_config

            # schema.yaml laden
            self._schema_config = read_yaml_file(self.config.schema_file_path)
        except Exception as e:
            # du kannst hier auch deine eigene Exception-Klasse nutzen
            raise Exception(e, sys)

    # ---------- Hilfsfunktionen ----------

    @staticmethod
    def read_data(file_path: Path | str) -> pd.DataFrame:
        """
        CSV -> DataFrame.
        """
        return pd.read_csv(file_path)
    
    

    def validate_number_of_columns(self, dataframe: pd.DataFrame) -> bool:
        """
        Prüft, ob die DataFrame-Spalten zur schema.yaml passen.

        schema.yaml:
            columns:
              - label
              - time_ms
              - Ax
              - Ay
              - Az
              - filename
        """
        try:
            required_columns = self._schema_config.get("columns", [])

            # columns kann Liste oder dict sein – wir erlauben beides
            if isinstance(required_columns, dict):
                required_columns = list(required_columns.keys())

            number_of_columns = len(required_columns)
            logging.info(f"[DataValidation] required #columns = {number_of_columns}")
            logging.info(f"[DataValidation] df has #columns = {len(dataframe.columns)}")

            # optional: check Namen, nicht nur Anzahl
            df_cols = list(dataframe.columns)
            if len(df_cols) != number_of_columns:
                logging.info(f"[DataValidation] Column count mismatch. df={df_cols}, schema={required_columns}")
                return False

            missing = [c for c in required_columns if c not in df_cols]
            if missing:
                logging.info(f"[DataValidation] Missing columns: {missing}")
                return False

            return True
        except Exception as e:
            raise Exception(e, sys)

    

    # ---------- Orchestrierung ----------

    def initiate_data_validation(self) -> DataValidationArtifact:
        """
        Orchestriert den kompletten Data-Validation-Schritt:

        1. ds_norm, ds_anom, ds_all (aus DataIngestionArtifact) laden.
        2. Spalten gegen schema.yaml validieren.
        3. KS-Drift-Test zwischen ds_norm (base) und ds_anom (current).
        4. validierte CSVs + drift_report.yaml speichern.
        5. DataValidationArtifact zurückgeben.
        """
        try:
            # Pfade aus DataIngestionArtifact
            norm_path = Path(self.data_ingestion_artifact.normal_file_path)
            anom_path = Path(self.data_ingestion_artifact.anomal_file_path)
            all_path = Path(self.data_ingestion_artifact.combined_file_path)

            # 1) CSVs laden (leere DataFrames, wenn Datei fehlt)
            ds_norm = self.read_data(norm_path) if norm_path.exists() else pd.DataFrame()
            ds_anom = self.read_data(anom_path) if anom_path.exists() else pd.DataFrame()
            ds_all = self.read_data(all_path) if all_path.exists() else pd.DataFrame()

            # 2) Spaltenvalidierung
            for name, df in [("normal", ds_norm), ("anomal", ds_anom), ("all", ds_all)]:
                if df.empty:
                    logging.info(f"[DataValidation] Warning: {name} DataFrame is empty.")
                    continue
                if not self.validate_number_of_columns(df):
                    raise ValueError(f"[DataValidation] {name} DataFrame does not match schema columns.")

            

            # 4) Validierte CSVs abspeichern
            os.makedirs(os.path.dirname(self.config.valid_norm_file_path), exist_ok=True)

            ds_norm.to_csv(self.config.valid_norm_file_path, index=False)
            ds_anom.to_csv(self.config.valid_anom_file_path, index=False)
            ds_all.to_csv(self.config.valid_all_file_path, index=False)

            # 5) Artefakt zurückgeben
            return DataValidationArtifact(
                
                valid_norm_file_path=Path(self.config.valid_norm_file_path),
                valid_anom_file_path=Path(self.config.valid_anom_file_path),
                valid_all_file_path=Path(self.config.valid_all_file_path),
                
            )
        except Exception as e:
            raise Exception(e, sys)
