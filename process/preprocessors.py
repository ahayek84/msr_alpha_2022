# description of the file : 
"""
ABI
1 - Data loading and imputation 
"""
import pandas as pd

def read_data():
    files = ["mongodb","react","socketio"]
    frames = []
    for file in files:
        df = pd.read_csv(f"./data/{file}.csv",sep=";",encoding="utf-8")
        df["source"] = [f"{file}" for _ in range(len(df))]
        frames.append(df)
    return pd.concat(frames)

if __name__ == "__main__":
    data = read_data()
    print(data.info())
