from credit_card.entity import config_entity,artifact_entity
from credit_card.logger import logging
from credit_card.exception import DefaultException
import os,sys
import pandas as pd
import numpy as np
from credit_card import utils
from sklearn.preprocessing import RobustScaler,PowerTransformer
from sklearn.pipeline import Pipeline
from credit_card.config import TARGET_COLUMN
from imblearn.combine import SMOTETomek
from sklearn.impute import SimpleImputer

NUMERICAL_COL=['LIMIT_BAL','AGE','BILL_AMT1','BILL_AMT2','BILL_AMT3','BILL_AMT4','BILL_AMT5','BILL_AMT6','PAY_AMT1','PAY_AMT2','PAY_AMT3','PAY_AMT4','PAY_AMT5','PAY_AMT6']

class DataTransformation:
    def __init__(self,data_transformation_config,data_ingestion_artifact):
        try:
            logging.info(f"{'>>'*20} Data Transformation{'<<'*20}")
            self.data_transformation_config=data_transformation_config
            self.data_ingestion_artifact=data_ingestion_artifact
        except Exception as e:
            raise DefaultException(e, sys)
    @classmethod
    def get_data_transformer_object(cls):
        try:
            #simple_imputer = SimpleImputer(strategy='constant', fill_value=0)
            robust_scaler =  RobustScaler()
            pipeline = Pipeline(steps=[('RobustScaler',robust_scaler)])
            return pipeline
            
        except Exception as e:
            raise DefaultException(e,sys)

    def apply_power_transform(self,df):
        try:
            for i in NUMERICAL_COL:
                transformer=PowerTransformer(method='yeo-johnson')
                transformer.fit(df[NUMERICAL_COL])
                df[NUMERICAL_COL]=transformer.transform(df[NUMERICAL_COL])
            return df
        except Exception as e:
            raise DefaultException(e,sys)


    def initiate_data_transformation(self):
        try:
            #reading training and testing file path
            train_df=pd.read_csv(self.data_ingestion_artifact.train_file_path)
            test_df=pd.read_csv(self.data_ingestion_artifact.test_file_path)

            logging.info(f"Before applying power transformation {train_df.head()}")

            train_df=self.apply_power_transform(train_df)
            test_df=self.apply_power_transform(test_df)

            logging.info(f"After applying power transformation {train_df.head()}")

            #selecting input feature for train and test df
            input_feature_train_df=train_df.drop(TARGET_COLUMN,axis=1)
            input_feature_test_df=test_df.drop(TARGET_COLUMN,axis=1)
            #input_feature_train_df.drop(['_id'],axis=1,inplace=True)
            #input_feature_test_df.drop(['_id'],axis=1,inplace=True)
            

            #selecting target feature for train and test dataframe
            target_feature_train_df = pd.DataFrame(train_df[TARGET_COLUMN])
            target_feature_test_df = pd.DataFrame(test_df[TARGET_COLUMN])

            transformation_pipeline = DataTransformation.get_data_transformer_object()
            transformation_pipeline.fit(input_feature_train_df)
            
            logging.info(f"Before standardisation {input_feature_train_df}")
            #transforming input features
            input_feature_train_arr = transformation_pipeline.transform(input_feature_train_df)
            input_feature_test_arr = transformation_pipeline.transform(input_feature_test_df)

            logging.info(f"After standardisation {input_feature_train_arr}")

            #rectify unbalanced dataset
            logging.info(f"Before Smote : {target_feature_train_df.value_counts()}")
            smt=SMOTETomek(random_state=0)
            input_feature_train_arr,target_feature_train_arr=smt.fit_resample(input_feature_train_arr,target_feature_train_df)

            logging.info(f"After smote : {target_feature_train_arr.value_counts()}")
            logging.info(f"unique values : {target_feature_train_arr['default.payment.next.month'].unique()}")

            logging.info(f"Before resampling in testing set Input: {input_feature_test_arr.shape} Target:{target_feature_test_df.shape}")
            input_feature_test_arr, target_feature_test_arr = smt.fit_resample(input_feature_test_arr, target_feature_test_df)
            logging.info(f"After resampling in testing set Input: {input_feature_test_arr.shape} Target:{target_feature_test_arr.shape}")
            

            #standardise the data
            #scaler=RobustScaler()
            #scaler.fit(input_feature_train_df)
            #input_feature_train_df=pd.DataFrame(scaler.transform(input_feature_train_df))
            #input_feature_test_df=pd.DataFrame(scaler.transform(input_feature_test_df))

            logging.info(f"input feature train arr {type(input_feature_train_arr)}")
            logging.info(f"input feature test arr {type(input_feature_test_arr)}")
            logging.info(f"input feature test arr values {input_feature_test_arr}")
            

            train_arr = pd.concat([pd.DataFrame(input_feature_train_arr), pd.DataFrame(target_feature_train_arr)],axis=1)
            test_arr = pd.concat([pd.DataFrame(input_feature_test_arr), pd.DataFrame(target_feature_test_arr)],axis=1)

            logging.info(f"transformed train path {self.data_transformation_config.transformed_train_file}")
            logging.info(f"transformed test path {self.data_transformation_config.transformed_test_file}")

            utils.save_df_to_file(file_path=self.data_transformation_config.transformed_train_file, df=train_arr)
            utils.save_df_to_file(file_path=self.data_transformation_config.transformed_test_file, df=test_arr)

            utils.save_object(file_path=self.data_transformation_config.transform_object_path,obj=transformation_pipeline)

            data_transformation_artifact=artifact_entity.DataTransformationArtifact(transform_object_path=self.data_transformation_config.transform_object_path,transformed_train_file=self.data_transformation_config.transformed_train_file, 
            transformed_test_file=self.data_transformation_config.transformed_test_file)

            logging.info(f"Data Transformation Artifact {data_transformation_artifact}")
            return data_transformation_artifact

        except Exception as e:
            raise DefaultException(e,sys)

            
