# src/components/model_trainer.py

from __future__ import annotations

import json
import os
import sys
from pathlib import Path
from typing import Dict

import joblib
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
    confusion_matrix,
    classification_report,
)
from sklearn.model_selection import GridSearchCV


from emlearn import convert as eml_convert

from src.entity.artifact_entity import (
    FeatureExtractionArtifact,
    ModelTrainerArtifact,
    ClassificationMetricArtifact,
)
from src.entity.config_entity import ModelTrainerConfig
from src.constants import pipeline_constants
from src.logging.logger import logging
from src.exception.exception import Exception

import mlflow
# wenn du Dagshub verwenden willst, kannst du das später ergänzen


class ModelTrainer:
    """
    Nutzt die von FeatureExtraction erzeugten npz-Dateien (train/test),
    trainiert einen RandomForest mit GridSearchCV,
    logged Metriken zu MLflow und exportiert das Modell nach C (emlearn).
    """

    def __init__(
        self,
        model_trainer_config: ModelTrainerConfig,
        feature_extraction_artifact: FeatureExtractionArtifact,
    ):
        self.cfg = model_trainer_config
        self.fe_artifact = feature_extraction_artifact

    # ----------------------------------------------------
    # Hilfsfunktionen
    # ----------------------------------------------------
    def _load_train_test(self):
        """
        Lädt train_features.npz und test_features.npz aus FeatureExtractionArtifact.
        Erwartet: np.savez(..., X=X, y=y)
        """
        train_npz = np.load(self.fe_artifact.train_features_file_path)
        test_npz = np.load(self.fe_artifact.test_features_file_path)

        X_train, y_train = train_npz["X"], train_npz["y"]
        X_test, y_test = test_npz["X"], test_npz["y"]

        logging.info(
            f"[ModelTrainer] Loaded train/test features: "
            f"X_train={X_train.shape}, X_test={X_test.shape}"
        )

        return X_train, y_train, X_test, y_test

    def _compute_metrics(
        self,
        y_true,
        y_pred,
        model: RandomForestClassifier,
        X_for_auc=None,
    ) -> ClassificationMetricArtifact:
        """
        Berechnet F1, Precision, Recall + optional ROC-AUC (wenn predict_proba vorhanden).
        """
        f1 = f1_score(y_true, y_pred)
        prec = precision_score(y_true, y_pred)
        rec = recall_score(y_true, y_pred)

        roc = None
        if (X_for_auc is not None) and hasattr(model, "predict_proba"):
            try:
                proba = model.predict_proba(X_for_auc)[:, 1]  # p(class=1 = normal)
                roc = roc_auc_score(y_true, proba)
            except Exception as e:
                logging.warning(f"[ModelTrainer] ROC-AUC konnte nicht berechnet werden: {e}")

        return ClassificationMetricArtifact(
            f1_score=f1,
            precision_score=prec,
            recall_score=rec,
            roc_auc=roc,
        )

    def _track_mlflow(
        self,
        model: RandomForestClassifier,
        best_params: Dict[str, object],
        train_metric: ClassificationMetricArtifact,
        test_metric: ClassificationMetricArtifact,
    ):
        """
        Loggt Hyperparameter, Metriken und Modell zu MLflow.
        Tracking-URI kannst du über Umgebung setzen (oder lokal lassen).
        """
        try:
            with mlflow.start_run(run_name="rf_gridsearch"):
                # Hyperparameter
                for k, v in best_params.items():
                    mlflow.log_param(k, v)

                # Train-Metriken
                mlflow.log_metric("train_f1", train_metric.f1_score)
                mlflow.log_metric("train_precision", train_metric.precision_score)
                mlflow.log_metric("train_recall", train_metric.recall_score)
                if train_metric.roc_auc is not None:
                    mlflow.log_metric("train_roc_auc", train_metric.roc_auc)

                # Test-Metriken
                mlflow.log_metric("test_f1", test_metric.f1_score)
                mlflow.log_metric("test_precision", test_metric.precision_score)
                mlflow.log_metric("test_recall", test_metric.recall_score)
                if test_metric.roc_auc is not None:
                    mlflow.log_metric("test_roc_auc", test_metric.roc_auc)

                # Modell selbst loggen
                mlflow.sklearn.log_model(model, "rf_model")
        except Exception as e:
            logging.warning(f"[ModelTrainer] MLflow-Logging fehlgeschlagen: {e}")

    def _train_random_forest(
        self,
        X_train,
        y_train,
        X_test,
        y_test,
    ):
        
        base_rf = RandomForestClassifier(
            class_weight="balanced",
            random_state=self.cfg.seed,
            n_jobs=-1,
        )

        # einfacher Grid über ein paar sinnvolle Parameter
        param_grid = {
            "n_estimators": [40, 60, 80],
            "max_depth": [6, 8, 10],
            "min_samples_leaf": [1, 2, 4],
            "max_features": ["sqrt", "log2"],
        }

        logging.info("[ModelTrainer] Starte GridSearchCV für RandomForest ...")

        gs = GridSearchCV(
            estimator=base_rf,
            param_grid=param_grid,
            cv=3,
            scoring="f1",   # F1-score für Binary-Klassifikation
            n_jobs=-1,
            verbose=1,
        )
        gs.fit(X_train, y_train)

        best_rf: RandomForestClassifier = gs.best_estimator_
        best_params = gs.best_params_

        logging.info(f"[ModelTrainer] Beste Parameter: {best_params}")

        # Predictions & Metriken
        y_train_pred = best_rf.predict(X_train)
        y_test_pred = best_rf.predict(X_test)

        train_metric = self._compute_metrics(
            y_true=y_train,
            y_pred=y_train_pred,
            model=best_rf,
            X_for_auc=X_train,
        )
        test_metric = self._compute_metrics(
            y_true=y_test,
            y_pred=y_test_pred,
            model=best_rf,
            X_for_auc=X_test,
        )

        # Optional: Confusion Matrix / Classification Report wie im Notebook ausgeben
        cm = confusion_matrix(y_test, y_test_pred, labels=[0, 1])
        logging.info(
            f"[ModelTrainer] Confusion (rows=true [0,1] anomal,normal):\n{cm}"
        )
        logging.info(
            "[ModelTrainer] Classification report (test):\n"
            + classification_report(y_test, y_test_pred, target_names=["anomalie", "normal"])
        )

        return best_rf, best_params, train_metric, test_metric

    def _export_emlearn_model(
        self,
        model: RandomForestClassifier,
        best_params: Dict[str, object],
    ):
        """
        Entspricht Notebook Cell 7: emlearn-Konvertierung + model_info.json
        """
        #codegen_dir = Path(self.cfg.codegen_dir)
        #codegen_dir.mkdir(parents=True, exist_ok=True)

        model_h_path = Path(self.cfg.emlearn_header_file_path)

        # emlearn-Konvertierung
        cmodel = eml_convert(
            model,
            method="inline",   # header-only
            dtype="float",
        )
        cmodel.save("rf_model", str(model_h_path))

        # Metadata (wie im Notebook)
        info = {
            "fs_hz": pipeline_constants.FS_HZ,
            "win": pipeline_constants.WIN,
            "hop": pipeline_constants.HOP,
            "feat_dim": pipeline_constants.FEAT_DIM,
            "class_order": model.classes_.tolist(),  # [0,1] -> [anomalie, normal]
            "best_params": best_params,
        }

        info_path = Path(self.cfg.model_info_json_path)
        with open(info_path, "w") as f:
            json.dump(info, f, indent=2)

        logging.info(f"[ModelTrainer] ✅ emlearn model.h geschrieben: {model_h_path}")
        logging.info(f"[ModelTrainer] ✅ model_info.json geschrieben: {info_path}")

        return model_h_path, info_path

    # ----------------------------------------------------
    # Public API: entspricht "initiate_model_trainer"
    # ----------------------------------------------------
    def initiate_model_trainer(self) -> ModelTrainerArtifact:
        """
        Orchestriert:
        - Load train/test features (npz)
        - Train RandomForest + GridSearchCV
        - Logge alles zu MLflow
        - Speichere sklearn-Modell (joblib)
        - Exportiere emlearn-Header + model_info.json
        - Liefere ModelTrainerArtifact zurück
        """
        try:
            X_train, y_train, X_test, y_test = self._load_train_test()

            best_model, best_params, train_metric, test_metric = self._train_random_forest(
                X_train, y_train, X_test, y_test
            )

            # einfache Overfitting-Check (optional)
            if (train_metric.f1_score - test_metric.f1_score) > self.cfg.overfitting_threshold:
                logging.warning(
                    f"[ModelTrainer] Achtung: mögliches Overfitting. "
                    f"Train F1={train_metric.f1_score:.3f}, Test F1={test_metric.f1_score:.3f}"
                )

            if test_metric.f1_score < self.cfg.expected_f1:
                logging.warning(
                    f"[ModelTrainer] Test-F1 ({test_metric.f1_score:.3f}) "
                    f"unter erwartetem Wert ({self.cfg.expected_f1:.3f})"
                )

            # MLflow-Tracking
            self._track_mlflow(
                model=best_model,
                best_params=best_params,
                train_metric=train_metric,
                test_metric=test_metric,
            )

            # sklearn-Modell speichern
            os.makedirs(os.path.dirname(self.cfg.trained_model_file_path), exist_ok=True)
            joblib.dump(best_model, self.cfg.trained_model_file_path)

            

            # emlearn-Export
            model_h_path, info_path = self._export_emlearn_model(best_model, best_params)

            # Artifact
            return ModelTrainerArtifact(
                trained_model_file_path=Path(self.cfg.trained_model_file_path),
                emlearn_header_file_path=model_h_path,
                model_info_json_path=info_path,
                train_metric=train_metric,
                test_metric=test_metric,
                best_params=best_params,
            )

        except Exception as e:
            logging.error(f"[ModelTrainer] Fehler in initiate_model_trainer: {e}")
            raise Exception(e, sys)
