from credit_card.exception import DefaultException
import os,sys
from credit_card.predictor import ModelResolver
import pandas as pd
from credit_card.utils import load_object

PREDICTION_DIR="prediction"

def start_batch_prediction(self,input_file_path):
    try:
        os.makedirs(PREDICTION_DIR,exist_ok=True)
        model_resolver=ModelResolver(model_registry="saved_models")
        df=pd.read_csv(input_file_path)

        transformer=load_object(file_path=model_resolver.get_latest_transformer_path())

        input_feature_names=list(transformer.feature_names_in)
        input_arr=transformer.transform(df[input_feature_names])

        model=model_resolver.get_latest_model_path()
        model.predict(input_arr)

        prediction_file_name = os.path.basename(input_file_path).replace(".csv",f"{datetime.now().strftime('%m%d%Y__%H%M%S')}.csv")
        prediction_file_path = os.path.join(PREDICTION_DIR,prediction_file_name)
        df.to_csv(prediction_file_path,index=False,header=True)
        return prediction_file_path
    except Exception as e:
        raise 