from credit_card.exception import DefaultException
from credit_card.predictor import ModelResolver
from credit_card.entity import config_entity,artifact_entity
from credit_card.utils import load_object
import os,sys
import pandas as pd
import numpy as np
from credit_card.config import TARGET_COLUMN
from credit_card.components.data_transformation import NUMERICAL_COL
from sklearn.metrics import f1_score
from credit_card.logger import logging
from credit_card.components import data_transformation

class ModelEvaluation:

    def __init__(self,model_evaluation_config,data_ingestion_artifact,data_transformation_artifact,model_trainer_artifact,data_transformation):
        try:
            self.model_evaluation_config=model_evaluation_config
            self.data_ingestion_artifact=data_ingestion_artifact
            self.data_transformation_artifact=data_transformation_artifact
            self.model_trainer_artifact=model_trainer_artifact
            self.model_resolver=ModelResolver()
            self.data_transformation=data_transformation
        except Exception as e:
            raise DefaultException(e,sys)

    def initiate_model_evaluation(self):

        try:
            latest_dir_path=self.model_resolver.get_latest_dir_path()
            if latest_dir_path is None:
                model_eval_artifact=artifact_entity.ModelEvaluationArtifact(is_model_accepted=True,improved_accuracy=None)
                logging.info(f"Model Evaluation artifact {model_eval_artifact}")
                return model_eval_artifact
                
            logging.info(f"Finding location of previous transformer and model")
            transformer_path=self.model_resolver.get_latest_transformer_path()
            model_path=self.model_resolver.get_latest_model_path()
            logging.info(f"transformer_path , model_path {transformer_path,model_path}")

            logging.info(f"Previous versions of transformer and model")
            transformer=load_object(file_path=transformer_path)
            model=load_object(file_path=model_path)
            logging.info(f"transformer {transformer}")
            logging.info(f"model {model}")

            logging.info(f"current transformer and model")
            current_transformer=load_object(self.data_transformation_artifact.transform_object_path)
            current_model=load_object(self.model_trainer_artifact.model_path)

            test_df=pd.read_csv(self.data_ingestion_artifact.test_file_path)
            y_true=test_df[TARGET_COLUMN]

         # accuracy using previous trained model
            input_feature_name=list(transformer.feature_names_in_)
            logging.info(f"Before power transform {test_df}")
            test_df=self.data_transformation.apply_power_transform(test_df)
            logging.info(f"After power transform {test_df}")
            input_arr=transformer.transform(test_df[input_feature_name])
            y_pred=model.predict(input_arr)
            previous_model_score=f1_score(y_true=y_true,y_pred=y_pred)
            logging.info(f"Accuracy of previous model{previous_model_score}")

        # accuracy using current trained model
            input_feature_name=list(current_transformer.feature_names_in_)
            test_df=self.data_transformation.apply_power_transform(test_df)
            input_arr=current_transformer.transform(test_df[input_feature_name])
            y_pred=current_model.predict(input_arr)
            current_model_score = f1_score(y_true=y_true, y_pred=y_pred)
            logging.info(f"Accuracy using current trained model: {current_model_score}")

            if current_model_score<=previous_model_score:
                logging.info(f"Current trained model is not better than previous model")
                raise Exception(f"Current trained model is not better than previous model")

            model_eval_artifact=artifact_entity.ModelEvaluationArtifact(is_model_accepted=True, improved_accuracy=current_model_score-previous_model_score)
            logging.info(f"Model Evaluation Artifact {model_eval_artifact}")
            return model_eval_artifact

        except Exception as e:
            raise DefaultException(e,sys)



