# description of the file : 
"""
ABI
1 - Data loading and imputation 
"""
import pandas as pd
import numpy as np
from utils import MultiColumnLabelEncoder
from mlsmote import MLSMOTE

def read_data():
    files = ["mongodb","react","socketio"]
    frames = []
    for file in files:
        df = pd.read_csv(f"./data/{file}.csv",sep=";",encoding="utf-8")
        df["source"] = [f"{file}" for _ in range(len(df))]
        frames.append(df)
    return pd.concat(frames)

def data_preprocessing(source="all"):
    data = read_data()

    if source in ["mongodb","react","socketio"]:
        data = data.loc[data.source == source,:]

    cols = ["login","name","email","source"]
    res = MultiColumnLabelEncoder(columns = cols).fit_transform(data)
    res = res.fillna(0)
    return res

def create_dataset(source="all"):
    all_data = data_preprocessing(source=source)
    all_data.to_excel(f"./data/all_data.xlsx")
    X = all_data.loc[:,all_data.columns != "survey"]
    y = [x[0] for x in all_data[["survey"]].values]
    X_res,y_res = MLSMOTE(X, y, 1)     #Applying MLSMOTE to augment the dataframe 
    return X_res,y_res


if __name__ == "__main__":
    X, y = create_dataset(source="react")  
    print(X.shape)
    #X.to_excel(f"./data/all_X_data.xlsx")
