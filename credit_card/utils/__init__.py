import pandas as pd
from credit_card.config import mongo_client
import logging
import os,sys
from credit_card.logger import logging
from credit_card.exception import DefaultException
import yaml
import numpy as np
import dill

def get_collection_as_dataframe(database_name:str,collection_name:str):
    try:
        logging.info(f"Reading data from database {database_name} and collection {collection_name}")
        df=pd.DataFrame(list(mongo_client[database_name][collection_name].find()))
        logging.info("Found columns {df.columns}")
        if "_id" in df.columns:
            logging.info("Removing ID column")
            df.drop(["_id"],axis=1,inplace=True)
            logging.info(f"Rows and columns in {df.shape}")
        return df
    except Exception as e:
        raise DefaultException(e, sys)

def write_yaml_file(file_path,data):
    try:
        file_dir=os.path.dirname(file_path)
        os.makedirs(file_dir,exist_ok=True)
        with open(file_path,"w") as file_writer:
            yaml.dump(data,file_writer)

    except Exception as e:
        raise DefaultException(e, sys)

def save_df_to_file(file_path,df):
    try:
        file_dir=os.path.dirname(file_path)
        os.makedirs(file_dir,exist_ok=True)
        df.to_csv(path_or_buf=file_path,index=False,header=True)
    except Exception as e:
        raise DefaultException(e,sys)

def load_df_from_file(file_path):
    try:
        df=pd.read_csv(file_path)
        return df
    except Exception as e:
        raise DefaultException(e,sys)

def save_object(file_path,obj):
    try:
        logging.info(f"Entered save_obj method in utils")
        os.makedirs(os.path.dirname(file_path),exist_ok=True)
        with open(file_path,"wb") as file_obj:
           dill.dump(obj,file_obj)
        logging.info(f"Exit save object method of utils")
    except Exception as e:
        raise DefaultException(e, sys)

def load_object(file_path,)->object:
    try:
        if not os.path.exists(file_path):
            raise Exception(f"The file path {file_path} does not exist")
        with open(file_path,"rb") as file_obj:
            return dill.load(file_obj)
    except Exception as e:
        raise DefaultException(e,sys) from e

