from credit_card.pipeline.training_pipeline import start_training_pipeline
#from credit_card.pipeline.batch_prediction import start_batch_prediction
from credit_card.entity import artifact_entity,config_entity
from credit_card.components.data_ingestion import DataIngestion
from credit_card.components.data_validation import DataValidation
from credit_card.components.data_transformation import DataTransformation
from credit_card.components.model_trainer import ModelTrainer
from credit_card.components.model_evaluation import ModelEvaluation
from credit_card.components.model_pusher import ModelPusher
import os,sys
from credit_card.pipeline.training_pipeline import start_training_pipeline



file_path="/config/workspace/aps_failure_training_set1.csv"
print(__name__)
if __name__=="__main__":
     try:
          start_training_pipeline()
          #output_file = start_batch_prediction(input_file_path=file_path)
          #print(output_file)
          

     except Exception as e:
          print(e)