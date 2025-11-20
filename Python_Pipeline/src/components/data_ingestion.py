from pathlib import Path
import pandas as pd
import sys
from typing import Union

from src.entity.artifact_entity import DataIngestionArtifact
from src.entity.config_entity import DataIngestionConfig

from src.logging.logger import logging
from src.exception.exception import Exception

class DataIngestion:
    """
    Implementiert die Notebook-Logik:

        ds_norm = load_folder(NORMAL_DIR, "normal")
        ds_anom = load_folder(ANOM_DIR, "anomalie")

    und speichert:
        - ds_norm  -> ds_norm_file_path
        - ds_anom  -> ds_anom_file_path
        - ds_all   -> ds_all_file_path
    """
 
    def __init__(self, config: DataIngestionConfig):
        try:
            self.cfg = config
        except Exception as e:
            raise Exception(e,sys)

    def _load_folder(self, folder: Union[str, Path], label: str) -> pd.DataFrame:
        """
        Exakte Umsetzung deines Notebook-load_folder(...)

        Gibt ein DataFrame mit Spalten:
            ['label', 'time_ms', 'Ax', 'Ay', 'Az', 'filename']
        zurück.
        """
        try:
            folder = Path(folder)
            rows = []
    
            for f in sorted(folder.glob("*.csv")):
                df = pd.read_csv(f)
    
                # exakt wie im Notebook
                df = df.rename(
                    columns={
                        "timestamp_ms": "time_ms",
                        "ax_raw": "Ax",
                        "ay_raw": "Ay",
                        "az_raw": "Az",
                        "label": "label",
                    }
                )
    
                keep = ["time_ms", "Ax", "Ay", "Az"]
                for k in keep:
                    if k not in df.columns:
                        raise ValueError(f"Column {k} missing in {f}")
    
                df["label"] = label
                df["filename"] = f.stem  # ohne .csv
    
                rows.append(df[["label", "time_ms", "Ax", "Ay", "Az", "filename"]])
    
            if rows:
                return pd.concat(rows, ignore_index=True)
    
            # leerer Default mit genau derselben Spaltenstruktur wie im Notebook
            return pd.DataFrame(columns=["label", "time_ms", "Ax", "Ay", "Az", "filename"])
        except Exception as e:
            raise Exception(e,sys)

    def initiate_data_ingestion(self) -> DataIngestionArtifact:
        """
        - erzeugt ds_norm, ds_anom wie im Notebook
        - speichert ds_norm, ds_anom, ds_all in artifacts/<ts>/data_ingestion
        - gibt ein DataIngestionArtifact mit Pfaden + Zeilenzahlen zurück
        """
        try:
            #logging.INFO("data Ingestion started")
            # Ziel-Ordner existieren lassen
            data_ingestion_dir = Path(self.cfg.data_ingestion_dir)
            data_ingestion_dir.mkdir(parents=True, exist_ok=True)
    
            # 1) ds_norm und ds_anom wie im Notebook erzeugen
            ds_norm = self._load_folder(self.cfg.normal_data_dir, self.cfg.normal_label)
            ds_anom = self._load_folder(self.cfg.anomal_data_dir, self.cfg.anomal_label)
    
            # 2) Output-Dateipfade aus Config
            norm_out = Path(self.cfg.ds_norm_file_path)
            anom_out = Path(self.cfg.ds_anom_file_path)
            all_out = Path(self.cfg.ds_all_file_path)
    
            # 3) Speichern (CSV)
            ds_norm.to_csv(norm_out, index=False)
            ds_anom.to_csv(anom_out, index=False)
    
            if not ds_norm.empty or not ds_anom.empty:
                ds_all = pd.concat([ds_norm, ds_anom], ignore_index=True)
            else:
                ds_all = pd.DataFrame(columns=["label", "time_ms", "Ax", "Ay", "Az", "filename"])
    
            ds_all.to_csv(all_out, index=False)
            #logging.INFO("Data Ingestion finished")
            # 4) Artifact zurückgeben
            return DataIngestionArtifact(
                normal_file_path=norm_out,
                anomal_file_path=anom_out,
                combined_file_path=all_out,
                
            )
        except Exception as e:
            raise Exception(e,sys)
          
    
