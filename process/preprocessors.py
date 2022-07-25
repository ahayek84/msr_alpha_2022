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
<<<<<<< HEAD
from sklearn.preprocessing import StandardScaler, MinMaxScaler
=======
>>>>>>> bbb57d94b59715f59e21cb3a85d496a4554e6826

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


def scaler_fit(X):
    scaler = MinMaxScaler()
    scaler.fit(X)
    return scaler


def create_dataset(source="all",use_smote=True):
    all_data = data_preprocessing(source=source)
    all_data.to_excel(f"../data/all_data.xlsx")
    X = all_data.loc[:,all_data.columns != "survey"]
    y = [x[0] for x in all_data[["survey"]].values]
    if use_smote:
        X_res,y_res = MLSMOTE(X, y, 1)     #Applying MLSMOTE to augment the dataframe 
    else:
        X_res,y_res = X,y
    X_train, X_test, y_train, y_test = split(X_res,y_res)     #Applying MLSMOTE to augment the dataframe 
    scaler = scaler_fit(X_train)
    X_train = scaler.transform(X_train)
    X_test = scaler.transform(X_test)
    return X_train, X_test, y_train, y_test


if __name__ == "__main__":
    X, y = create_dataset(source="react",use_smote=False)  
    print(X.shape)
    #X.to_excel(f"./data/all_X_data.xlsx")
