import sys
from src.pipeline.training_pipeline import TrainingPipeline
from src.logging.logger import logging
from src.exception.exception import Exception



try:
        train_pipeline=TrainingPipeline()
        train_pipeline.run_pipeline()
        print("training successfull. Pipeline finisched")
        logging.info("training successfull. Pipeline finisched")
except Exception as e:
        raise Exception(e,sys)