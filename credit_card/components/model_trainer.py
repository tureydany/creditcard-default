from credit_card.entity import config_entity,artifact_entity
from sklearn.ensemble import RandomForestClassifier
from credit_card.exception import DefaultException
from credit_card.logger import logging
from credit_card import utils 
from sklearn.metrics import f1_score
import os,sys
import pandas as pd
import numpy as np


class ModelTrainer:
    def __init__(self,model_trainer_config:config_entity.ModelTrainerConfig,data_transformation_artifact:artifact_entity.DataTransformationArtifact):
        try:
            self.model_trainer_config=model_trainer_config
            self.data_transformation_artifact=data_transformation_artifact
        except Exception as e:
            raise DefaultException(e,sys)

    def train_model(self,x,y):
        try:
            rfc=RandomForestClassifier()
            rfc.fit(x,y)
            return rfc

        except Exception as e:
            raise DefaultException(e,sys)

    def initiate_model_training(self,):
        try:
            
            train_arr=pd.read_csv(self.data_transformation_artifact.transformed_train_file)
            test_arr=pd.read_csv(self.data_transformation_artifact.transformed_test_file)

            logging.info(f"train_arr {train_arr}")

            x_train=train_arr.drop(columns=['default.payment.next.month'])
            y_train=train_arr['default.payment.next.month']
            x_test=test_arr.drop(columns=['default.payment.next.month'])
            y_test=test_arr['default.payment.next.month']

            model=self.train_model(x=x_train,y=y_train)

            y_pred_train=model.predict(x_train)
            f1_train_score=f1_score(y_true=y_train,y_pred=y_pred_train)

            y_pred_test=model.predict(x_test)
            f1_test_score=f1_score(y_true=y_test,y_pred=y_pred_test)

            logging.info(f"train score:{f1_train_score} and tests score {f1_test_score}")

            logging.info(f"Checking if our model is underfitting or not")
            if f1_test_score<self.model_trainer_config.expected_score:
                raise Exception(f"Model is not good as it is not able to give \
                expected accuracy: {self.model_trainer_config.expected_score}: model actual score: {f1_test_score}")

            logging.info(f"Checking if model is overfitting or not")
            diff=abs(f1_train_score-f1_test_score)
            if diff>self.model_trainer_config.overfitting_threshold:
                raise Exception(f"Train and test score diff {diff} is more than overfitting threshold {self.model_trainer_config.overfitting_threshold}")

            utils.save_object(file_path=self.model_trainer_config.model_path, obj=model)

            logging.info(f"Preparing model trainer artifact")
            model_trainer_artifact=artifact_entity.ModelTrainerArtifact(model_path=self.model_trainer_config.model_path, f1_train_score=f1_train_score, f1_test_score=f1_test_score)
            logging.info(f"Model Trainer Artifact {model_trainer_artifact}")
            return model_trainer_artifact
        except Exception as e:
            raise DefaultException(e,sys)