# description of the file : 
"""
ABI
1 - Data loading and imputation 
"""
import pandas as pd
import numpy as np
from utils import get_newlabel
from mlsmote import MLSMOTE
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
import warnings
warnings.filterwarnings('always')

def read_data():
    files = ["mongodb","react","socketio"]
    frames = []
    for file in files:
        df = pd.read_csv(f"../data/{file}.csv",sep=";",encoding="utf-8")
        df["source"] = [f"{file}" for _ in range(len(df))]
        frames.append(df)
    return pd.concat(frames)


import math
def imputedaysBetweenImports(row):
    if math.isnan (row.daysBetweenImports):
        if (row.imports == 1):
            return 0
        if (row.imports == 0):
            return -1
    else:
        return row.daysBetweenImports
        


def data_imputation(data):
    """
    daysSinceFirstImport and daysSinceLastImport == 0
    avgDaysCommitsImportLibrary = maximal observed value
    daysBetweenImports = Therefore, we assigned a zero value when
    imports = 1, and −1 when imports = 0.
    """
    
    data['daysSinceFirstImport'] = data['daysSinceFirstImport'].fillna(0)
    data['daysSinceLastImport'] = data['daysSinceLastImport'].fillna(0)
    data['avgDaysCommitsImportLibrary'] = data['avgDaysCommitsImportLibrary'].fillna(np.max(data['avgDaysCommitsImportLibrary'].values))
    data["daysBetweenImports"] = data.apply(imputedaysBetweenImports, axis=1)
    return data.fillna(-99999)


def data_preprocessing(combine_labels,source="all"):
    data = read_data()
    data = drop_cols(data)
    if combine_labels:
        data["survey"] = data.apply(get_newlabel,axis=1)

    if source in ["mongodb","react","socketio"]:
        data = data.loc[data.source == source,:]
    source_idx = {"mongodb":0,"react":1,"socketio":2}

    data["source"] = data.apply(lambda x: source_idx[x["source"]],axis=1)
    data = data_imputation(data)
    return data


def drop_cols(df,cols=[]):
    """
    login,name,email --> Not relevant for prediction
    commitsImportLibrary,projectsImport ---> highly correlated feature
    """
    drop = ["login","name","email","commitsImportLibrary","projectsImport"]
    
    if len(cols) != 0:
        drop += cols
    return df.drop(columns=drop)


def split(X,y,test_size=0.2):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size,random_state=42,stratify=y)
    return X_train, X_test, y_train, y_test


def scaler_fit(X):
    scaler = MinMaxScaler()
    scaler.fit(X)
    return scaler


def create_dataset(source="all", use_smote=True, combine_labels=False):
    all_data = data_preprocessing(combine_labels, source=source)
    X = all_data.loc[:,all_data.columns != "survey"]
    y = [x[0] for x in all_data[["survey"]].values]
    X_train, X_test, y_train, y_test = split(X,y) 
    if use_smote:
        X_train,y_train = MLSMOTE(X_train, y_train)     #Applying MLSMOTE to augment the dataframe 
    test_source = X_test[["source"]].copy()
    scaler = scaler_fit(X_train)
    X_train = scaler.transform(X_train)
    X_test = scaler.transform(X_test)
    return X_train, X_test, y_train, y_test, test_source


if __name__ == "__main__":
    X_train, X_test, y_train, y_test, test_source = create_dataset(source="react",use_smote=False)
