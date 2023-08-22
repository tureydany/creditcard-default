from credit_card.entity import config_entity,artifact_entity
from credit_card.exception import DefaultException
from credit_card.logger import logging
from credit_card import utils
import os,sys
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split


class DataIngestion:

    def __init__(self,data_ingestion_config:config_entity.DataIngestionConfig):
        try:
            logging.info(f"{'>>'*20} Data Ingestion {'<<'*20}")
            self.data_ingestion_config=data_ingestion_config
        except Exception as e:
            raise DefaultException(e,sys)

    def initiate_data_ingestion(self)->artifact_entity.DataIngestionArtifact:
        try:

            logging.info("Exporting data as dataframe")
            df:pd.DataFrame=utils.get_collection_as_dataframe(database_name=self.data_ingestion_config.database_name,collection_name=self.data_ingestion_config.collection_name)
            logging.info(f"DataFrame {df}")

            #df.replace(to_replace=np.NaN,value=np.NAN,inplace=True)

            logging.info("Creating feature store dir")
            feature_store_dir=os.path.dirname(self.data_ingestion_config.feature_store_file_path)
            os.makedirs(feature_store_dir,exist_ok=True)

            logging.info("Saving data in feature store")
            df.to_csv(path_or_buf=self.data_ingestion_config.feature_store_file_path,index=False,header=True)

            #split data into train and test data
            logging.info("Split data into train and test ")
            train_df,test_df=train_test_split(df,test_size=self.data_ingestion_config.test_size)

            logging.info("Create dataset dir")
            dataset_dir=os.path.dirname(self.data_ingestion_config.train_file_path)
            os.makedirs(dataset_dir,exist_ok=True)

            logging.info("Save train and test data")
            train_df.to_csv(path_or_buf=self.data_ingestion_config.train_file_path,index=False,header=True)
            test_df.to_csv(path_or_buf=self.data_ingestion_config.test_file_path,index=False,header=True)

            data_ingestion_artifact=artifact_entity.DataIngestionArtifact(feature_store_file_path=self.data_ingestion_config.feature_store_file_path,
             train_file_path=self.data_ingestion_config.train_file_path, test_file_path=self.data_ingestion_config.test_file_path)

            logging.info(f"Data ingestion artifact: {data_ingestion_artifact}")
            return data_ingestion_artifact


        except Exception as e:
            raise DefaultException(e,sys)


