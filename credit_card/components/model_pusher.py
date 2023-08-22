from credit_card.entity.config_entity import ModelPusherConfig
from credit_card.entity.artifact_entity import DataTransformationArtifact,ModelTrainerArtifact,ModelPusherArtifact
from credit_card.exception import DefaultException
from credit_card.logger import logging
from credit_card.utils import load_object,save_object
import os,sys
from credit_card.predictor import ModelResolver
from credit_card.entity import config_entity,artifact_entity

class ModelPusher:

    def __init__(self,model_pusher_config:ModelPusherConfig,data_transformation_artifact,model_trainer_artifact:ModelTrainerArtifact):
        try:
            self.model_pusher_config=model_pusher_config
            self.data_transformation_artifact=data_transformation_artifact
            self.model_trainer_artifact=model_trainer_artifact
            self.model_resolver=ModelResolver(model_registry=self.model_pusher_config.saved_model_dir)
        except Exception as e :
            raise DefaultException(e,sys)

    def initiate_model_pusher(self):
        try:

            logging.info("Load Transformer and model")
            transformer=load_object(file_path=self.data_transformation_artifact.transform_object_path)
            model=load_object(file_path=self.model_trainer_artifact.model_path)

            #model pusher dir
            logging.info(f"Saving model into model pusher dir")
            save_object(file_path=self.model_pusher_config.pusher_transformer_path,obj=transformer)
            save_object(file_path=self.model_pusher_config.pusher_model_path, obj=model)

            #saved model dir
            logging.info(f"Saving model into saved model dir")
            transformer_path=self.model_resolver.get_latest_save_transformer_path()
            model_path=self.model_resolver.get_latest_save_model_path()

            save_object(file_path=transformer_path, obj=transformer)
            save_object(file_path=model_path, obj=model)

            model_pusher_artifact=artifact_entity.ModelPusherArtifact(pusher_model_dir=self.model_pusher_config.pusher_model_dir
            ,saved_model_dir=self.model_pusher_config.saved_model_dir)

            logging.info(f"Model pusher artifact: {model_pusher_artifact}")

            return model_pusher_artifact

        except Exception as e:
            raise DefaultException(e,sys)




    