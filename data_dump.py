import pymongo
import pandas as pd
import json
# Provide the mongodb localhost url to connect python to mongodb.
client = pymongo.MongoClient("mongodb+srv://turey_dany:arul2304@cluster0.egy97.mongodb.net/?retryWrites=true&w=majority")

DATABASE_NAME="defaulter"
Collection_name="creditcard"

if __name__=="__main__":
    df=pd.read_csv("/config/workspace/UCI_Credit_Card.csv")
    print(f"Rws and cls : {df.shape}")

    df.reset_index(drop=True,inplace=True)
    json_record=list(json.loads(df.T.to_json()).values())


    client[DATABASE_NAME][Collection_name].insert_many(json_record)
