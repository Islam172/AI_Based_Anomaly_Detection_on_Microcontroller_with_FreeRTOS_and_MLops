import os
import sys

from src.exception.exception import Exception
from src.logging.logger import logging

from src.components.data_ingestion import DataIngestion
from src.components.data_validation import DataValidation
from src.components.feature_extraction import FeatureExtraction
from src.components.model_trainer import ModelTrainer

from src.entity.config_entity import(
    TrainingPipelineConfig,
    DataIngestionConfig,
    DataValidationConfig,
    FeatureExtractionConfig,
    ModelTrainerConfig,
)

from src.entity.artifact_entity import (
    DataIngestionArtifact,
    DataValidationArtifact,
    FeatureExtractionArtifact,
    ModelTrainerArtifact,
)




class TrainingPipeline:
    """
    Class to orchestrate the machine learning pipeline, including data ingestion,
    validation, feature extraction and model training.
    """
    def __init__(self):
        """
        Initializes the training pipeline with configurations 
        """
        self.training_pipeline_config = TrainingPipelineConfig()  # Load pipeline configurations.
        

    def start_data_ingestion(self):
        """
        Initiates the data ingestion process to fetch and prepare raw data.

        Returns:
            DataIngestionArtifact: Contains paths to ingested train and test data.
        """
        try:
            self.data_ingestion_config = DataIngestionConfig(training_pipeline_config=self.training_pipeline_config)
            logging.info("Start data ingestion process")
            
            data_ingestion = DataIngestion(self.data_ingestion_config)
            data_ingestion_artifact = data_ingestion.initiate_data_ingestion()
            
            logging.info(f"Data ingestion completed. Artifact: {data_ingestion_artifact}")
            return data_ingestion_artifact
        except Exception as e:
            raise Exception(e, sys)
        
    def start_data_validation(self, data_ingestion_artifact: DataIngestionArtifact):
        """
        Initiates the data validation process to check data quality and schema.

        Parameters:
            data_ingestion_artifact (DataIngestionArtifact): Paths to ingested train and test data.

        Returns:
            DataValidationArtifact: Contains paths to validated data and validation reports.
        """
        try:
            data_validation_config = DataValidationConfig(self.training_pipeline_config)
            
            data_validation = DataValidation(
                data_ingestion_artifact=data_ingestion_artifact,
                data_validation_config=data_validation_config
            )
            
            logging.info("Initiate data validation process")
            data_validation_artifact = data_validation.initiate_data_validation()
            
            return data_validation_artifact
        except Exception as e:
            raise Exception(e, sys)
     
    def start_feature_extraction(self, data_validation_artifact: DataValidationArtifact):
        """
        Transforms the validated data into a format suitable for machine learning models.

        Parameters:
            data_validation_artifact (DataValidationArtifact): Paths to validated train and test data.

        Returns:
            DataTransformationArtifact: Contains paths to transformed train and test data.
        """
        try:
            feature_extraction_config = FeatureExtractionConfig(self.training_pipeline_config)
            
            feature_extraction = FeatureExtraction(
                feature_extraction_config,
                data_validation_artifact
                
            )
            
            feature_extraction_artifact = feature_extraction.initiate_feature_extraction()
            
            return feature_extraction_artifact
        except Exception as e:
            raise Exception(e, sys)
        
    def start_model_trainer(self, feature_extraction_artifact: FeatureExtractionArtifact) -> ModelTrainerArtifact:
        """
        Trains machine learning models using transformed data.

        Parameters:
            data_transformation_artifact (DataTransformationArtifact): Paths to transformed train and test data.

        Returns:
            ModelTrainerArtifact: Contains paths to the trained model and performance metrics.
        """
        try:
            self.model_trainer_config = ModelTrainerConfig(self.training_pipeline_config)
            
            model_trainer = ModelTrainer(
                
                self.model_trainer_config,
                feature_extraction_artifact
            )
            
            model_trainer_artifact = model_trainer.initiate_model_trainer()
            
            return model_trainer_artifact
        except Exception as e:
            raise Exception(e, sys)

     

    def run_pipeline(self):
        """
        Executes the entire pipeline: ingestion, validation, transformation, training, and syncing artifacts.

        Returns:
            ModelTrainerArtifact: The result of the trained model, including its metrics.
        """
        try:
            # Step 1: Data Ingestion
            data_ingestion_artifact = self.start_data_ingestion()
            
            # Step 2: Data Validation
            data_validation_artifact = self.start_data_validation(data_ingestion_artifact=data_ingestion_artifact)
            
            # Step 3: Data Transformation
            feature_extraction_artifact = self.start_feature_extraction(data_validation_artifact=data_validation_artifact)
            
            # Step 4: Model Training
            model_trainer_artifact = self.start_model_trainer(feature_extraction_artifact=feature_extraction_artifact)
            
            
            return model_trainer_artifact
        except Exception as e:
            raise Exception(e, sys)
      
      
if __name__ == "__main__":
    try:
        train_pipeline=TrainingPipeline()
        train_pipeline.run_pipeline()
        print("training successfull. Pipeline finisched")
        logging.info("training successfull. Pipeline finisched")
    except Exception as e:
        raise Exception(e,sys)