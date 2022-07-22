# description of the file : 
"""
ABI
1 - Data loading and imputation 
"""
import pandas as pd
import numpy as np
from utils import MultiColumnLabelEncoder
from mlsmote import MLSMOTE
from sklearn.model_selection import train_test_split

def read_data():
    files = ["mongodb","react","socketio"]
    frames = []
    for file in files:
        df = pd.read_csv(f"../data/{file}.csv",sep=";",encoding="utf-8")
        df["source"] = [f"{file}" for _ in range(len(df))]
        frames.append(df)
    return pd.concat(frames)

def data_preprocessing(source="all"):
    data = read_data()
    data = drop_cols(data)

    if source in ["mongodb","react","socketio"]:
        data = data.loc[data.source == source,:]

    cols = ["source"]
    res = MultiColumnLabelEncoder(columns = cols).fit_transform(data)
    res = res.fillna(0)
    return res


def drop_cols(df,cols=["login","name","email"]):
    if isinstance(cols,str):
        cols = [cols]
    return df.drop(columns=cols)


def split(X,y,test_size=0.2):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size)
    return X_train, X_test, y_train, y_test


def create_dataset(source="all"):
    all_data = data_preprocessing(source=source)
    all_data.to_excel(f"../data/all_data.xlsx")
    X = all_data.loc[:,all_data.columns != "survey"]
    y = [x[0] for x in all_data[["survey"]].values]
    X_res,y_res = MLSMOTE(X, y, 1)
    X_train, X_test, y_train, y_test = split(X_res,y_res)     #Applying MLSMOTE to augment the dataframe 
    return X_train, X_test, y_train, y_test


if __name__ == "__main__":
    X, y = create_dataset(source="react")  
    print(X.shape)
    #X.to_excel(f"./data/all_X_data.xlsx")
