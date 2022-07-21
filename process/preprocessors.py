# description of the file : 
"""
ABI
1 - Data loading and imputation 
"""
import pandas as pd
from utils import MultiColumnLabelEncoder

def read_data():
    files = ["mongodb","react","socketio"]
    frames = []
    for file in files:
        df = pd.read_csv(f"./data/{file}.csv",sep=";",encoding="utf-8")
        df["source"] = [f"{file}" for _ in range(len(df))]
        frames.append(df)
    return pd.concat(frames)

def data_preprocessing():
    data = read_data()
    cols = ["login","name","email","source"]
    res = MultiColumnLabelEncoder(columns = cols).fit_transform(data)
    res = res.fillna(0)
    return res

if __name__ == "__main__":
    all_data = data_preprocessing()
    print(all_data.info())
    #res.to_excel(f"./data/all_data.xlsx")
