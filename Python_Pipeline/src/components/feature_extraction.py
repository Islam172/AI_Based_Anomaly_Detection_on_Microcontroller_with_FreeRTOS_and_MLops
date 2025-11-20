# src/components/feature_extraction.py

from __future__ import annotations

from pathlib import Path
from typing import List, Tuple

import os
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

from src.entity.artifact_entity import DataValidationArtifact, FeatureExtractionArtifact
from src.entity.config_entity import FeatureExtractionConfig
from src.constants import pipeline_constants
from src.exception.exception import Exception


class FeatureExtraction:
    """
    Reproduziert die Notebook-Logik (Cells 2–4 + Cell 6) auf Basis der
    validierten CSVs aus der DataValidation-Stage.

    Notebook-Referenz:
    ------------------
    # Axis-wise stats (nur NORMAL)
    AXES = ['Ax','Ay','Az']
    axis_mean = ds_norm[AXES].astype(np.float64).mean(axis=0).values
    axis_std  = ds_norm[AXES].astype(np.float64).std(axis=0, ddof=0).values
    axis_std  = np.where(axis_std==0.0, 1e-6, axis_std)

    # zscore_df_raw(df)
    # frame_indices(...)
    # features_from_window(...)
    # featurize_by_file_axis_z(...)
    # Xn, Xa, y, train_test_split(...)
    # axis_scaler.h schreiben
    """

    AXES = ["Ax", "Ay", "Az"]

    def __init__(
        self,
        config: FeatureExtractionConfig,
        data_validation_artifact: DataValidationArtifact,
    ):
        self.cfg = config
        self.dv_artifact = data_validation_artifact

        # Aus pipeline_constants (entspricht Notebook-Cell 0)
        self.WIN = pipeline_constants.WIN
        self.HOP = pipeline_constants.HOP
        self.FEAT_DIM = pipeline_constants.FEAT_DIM
        self.seed = pipeline_constants.GLOBAL_SEED

    # ------------------------------------------------------------------
    # 1) Axis-wise Normalization (Notebook: axis_mean / axis_std)
    # ------------------------------------------------------------------
    def _compute_axis_stats(self, df_norm: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray]:
        """
        Genau wie im Notebook:

            AXES = ['Ax','Ay','Az']
            axis_mean = ds_norm[AXES].astype(np.float64).mean(axis=0).values
            axis_std  = ds_norm[AXES].astype(np.float64).std(axis=0, ddof=0).values
            axis_std  = np.where(axis_std==0.0, 1e-6, axis_std)
        """
        axes = self.AXES
        arr = df_norm[axes].astype(np.float64)

        axis_mean = arr.mean(axis=0).values  # shape (3,)
        axis_std = arr.std(axis=0, ddof=0).values  # shape (3,)
        axis_std = np.where(axis_std == 0.0, 1e-6, axis_std)

        return axis_mean, axis_std

    def _zscore_df_raw(
        self, df: pd.DataFrame, axis_mean: np.ndarray, axis_std: np.ndarray
    ) -> pd.DataFrame:
        """
        1:1 wie Notebook zscore_df_raw(df):

            Z = df[AXES].astype(np.float64).values
            Z = (Z - axis_mean) / axis_std
            out = df.copy()
            out[AXES] = Z.astype(np.float32)
            return out
        """
        axes = self.AXES
        Z = df[axes].astype(np.float64).values
        Z = (Z - axis_mean) / axis_std
        out = df.copy()
        out[axes] = Z.astype(np.float32)
        return out

    # ------------------------------------------------------------------
    # 2) Windowing + Feature-Berechnung (frame_indices, features_from_window)
    # ------------------------------------------------------------------
    def _frame_indices(self, n: int):
        """
        Genau wie:

            def frame_indices(n, win=WIN, hop=HOP):
                i = 0
                while i + win <= n:
                    yield i, i+win
                    i += hop
        """
        i = 0
        while i + self.WIN <= n:
            yield i, i + self.WIN
            i += self.HOP

    def _features_from_window(self, win3: np.ndarray) -> np.ndarray:
        """
        (WIN,3) -> (6,)  [mean & std pro Achse]
        1:1 wie Notebook:

            X = []
            for k in range(3):
                x = win3[:, k].astype(np.float32)
                mu = float(np.mean(x))
                sd = float(np.std(x, ddof=0))
                X.extend([mu, sd])
            return np.array(X, dtype=np.float32)
        """
        X: List[float] = []
        for k in range(3):
            x = win3[:, k].astype(np.float32)
            mu = float(np.mean(x))
            sd = float(np.std(x, ddof=0))
            X.extend([mu, sd])
        return np.array(X, dtype=np.float32)

    def _featurize_by_file_axis_z(
        self,
        df: pd.DataFrame,
        axis_mean: np.ndarray,
        axis_std: np.ndarray,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Entspricht:

            def featurize_by_file_axis_z(df):
                feats, labels = [], []
                for fname, part in df.groupby('filename', sort=False):
                    partz = zscore_df_raw(part)
                    arr = partz[['Ax','Ay','Az']].values
                    for i0, i1 in frame_indices(len(arr)):
                        win = arr[i0:i1]
                        feats.append(features_from_window(win))
                        labels.append(part['label'].iloc[0])
                X = np.vstack(feats).astype(np.float32) if feats else ...
                y = np.array(labels)
        """
        feats: List[np.ndarray] = []
        labels: List[str] = []

        for fname, part in df.groupby("filename", sort=False):
            partz = self._zscore_df_raw(part, axis_mean, axis_std)
            arr = partz[self.AXES].values  # shape (N,3)

            for i0, i1 in self._frame_indices(len(arr)):
                win = arr[i0:i1]  # (WIN,3)
                feats.append(self._features_from_window(win))
                labels.append(part["label"].iloc[0])

        if feats:
            X = np.vstack(feats).astype(np.float32)
        else:
            X = np.zeros((0, self.FEAT_DIM), dtype=np.float32)

        y = np.array(labels)
        return X, y

    # ------------------------------------------------------------------
    # 3) Orchestrierung wie Notebook-Cells 2–4 + Cell 6
    # ------------------------------------------------------------------
    def initiate_feature_extraction(self) -> FeatureExtractionArtifact:
        """
        Hauptmethode:

        - lädt valid_norm_file_path & valid_anom_file_path (RAW)
        - berechnet axis_mean / axis_std aus NORMAL (wie Cell 2)
        - extrahiert Features Xn/Xa (wie Cell 3)
        - baut X, y (1=normal, 0=anomalie) + train/test split (Cell 4)
        - speichert:
            * axis_stats.npz
            * train_features.npz
            * test_features.npz
            * axis_scaler.h (Cell 6)
        - gibt FeatureExtractionArtifact zurück
        """

        # --------------------------
        # 1) Validierte CSVs laden
        # --------------------------
        norm_path: Path = self.dv_artifact.valid_norm_file_path
        anom_path: Path = self.dv_artifact.valid_anom_file_path
        # all_path: Path  = self.dv_artifact.valid_all_file_path  # aktuell in dieser Stage nicht genutzt

        df_norm = pd.read_csv(norm_path)
        df_anom = pd.read_csv(anom_path)

        required_cols = ["label", "time_ms", "Ax", "Ay", "Az", "filename"]
        for df, name in [
            (df_norm, "valid_norm_file_path"),
            (df_anom, "valid_anom_file_path"),
        ]:
            missing = [c for c in required_cols if c not in df.columns]
            if missing:
                raise ValueError(f"[FeatureExtraction] Missing columns {missing} in {name}")

        if df_norm.empty:
            raise RuntimeError("[FeatureExtraction] df_norm is empty – cannot compute axis stats.")

        # ---------------------------------------
        # 2) Axis-wise Statistik (nur NORMAL)
        # ---------------------------------------
        axis_mean, axis_std = self._compute_axis_stats(df_norm)

        # 2a) Axis-Stats als npz speichern (für weitere Analysen)
        axis_stats_path = Path(self.cfg.axis_stats_file_path)
        axis_stats_path.parent.mkdir(parents=True, exist_ok=True)
        np.savez(axis_stats_path, axis_mean=axis_mean, axis_std=axis_std)

        # 2b) axis_scaler.h für C/MCU exportieren (Notebook Cell 6)
        axis_scaler_path = Path(self.cfg.axis_scaler_header_path)
        axis_scaler_path.parent.mkdir(parents=True, exist_ok=True)
        #axis_h = codegen_dir / "axis_scaler.h"

        with open(axis_scaler_path, "w") as f:
            f.write("#pragma once\n#include <stdint.h>\n\n")
            f.write("/* Autogenerated: axis-wise z-score on raw Ax,Ay,Az */\n")
            f.write(
                "static const float AXIS_MEAN[3] = { %.9g, %.9g, %.9g };\n"
                % (axis_mean[0], axis_mean[1], axis_mean[2])
            )
            f.write(
                "static const float AXIS_STD[3]  = { %.9g, %.9g, %.9g };\n"
                % (axis_std[0], axis_std[1], axis_std[2])
            )

        # ---------------------------------------
        # 3) Feature-Extraktion für Normal/Anomal
        # ---------------------------------------
        Xn, yn_str = self._featurize_by_file_axis_z(df_norm, axis_mean, axis_std)
        Xa, ya_str = self._featurize_by_file_axis_z(df_anom, axis_mean, axis_std)

        # y-Labels in ints mappen: 1 = normal, 0 = anomalie
        y_norm = np.ones(len(Xn), dtype=int)
        y_anom = np.zeros(len(Xa), dtype=int)

        # ---------------------------------------
        # 4) Kombiniertes Dataset X, y (Cell 4)
        # ---------------------------------------
        if len(Xn) == 0 and len(Xa) == 0:
            raise RuntimeError("[FeatureExtraction] No windows produced from input data.")

        if len(Xa) > 0:
            X = np.vstack([Xn, Xa]).astype(np.float32)
            y = np.concatenate([y_norm, y_anom])
        else:
            # nur Normaldaten (z.B. reiner One-Class-Fall)
            X = Xn.astype(np.float32)
            y = y_norm

        # stratify nur, wenn beide Klassen vorkommen
        unique_classes = np.unique(y)
        stratify = y if len(unique_classes) > 1 else None

        # ---------------------------------------
        # 5) Train/Test-Split (wie Notebook)
        # ---------------------------------------
        X_train, X_test, y_train, y_test = train_test_split(
            X,
            y,
            test_size=self.cfg.test_size,
            random_state=self.cfg.seed,
            stratify=stratify,
        )

        # ---------------------------------------
        # 6) Speichern als npz-Artefakte
        # ---------------------------------------
        train_feat_path = Path(self.cfg.train_features_file_path)
        test_feat_path = Path(self.cfg.test_features_file_path)

        train_feat_path.parent.mkdir(parents=True, exist_ok=True)

        np.savez(train_feat_path, X=X_train, y=y_train)
        np.savez(test_feat_path, X=X_test, y=y_test)

        # ---------------------------------------
        # 7) Artifact zurückgeben
        # ---------------------------------------
        return FeatureExtractionArtifact(
            axis_stats_file_path=axis_stats_path,
            axis_scaler_header_path=axis_scaler_path,
            train_features_file_path=train_feat_path,
            test_features_file_path=test_feat_path,
            n_train_windows=X_train.shape[0],
            n_test_windows=X_test.shape[0],
        )
