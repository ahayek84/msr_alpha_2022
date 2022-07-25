# description of the file : 
"""
ABI
helper functions 
like SMOTE (upsample)  
"""
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.pipeline import Pipeline


class MultiColumnLabelEncoder:
    def __init__(self,columns = None):
        self.columns = columns # array of column names to encode

    def fit(self,X,y=None):
        return self # not relevant here

    def transform(self,X):
        '''
        Transforms columns of X specified in self.columns using
        LabelEncoder(). If no columns specified, transforms all
        columns in X.
        '''
        output = X.copy()
        if self.columns is not None:
            for col in self.columns:
                output[col] = LabelEncoder().fit_transform(output[col])
        else:
            for colname,col in output.iteritems():
                output[colname] = LabelEncoder().fit_transform(col)
        return output

    def fit_transform(self,X,y=None):
        return self.fit(X,y).transform(X)

def get_newlabel(row):
    survey = row["survey"]
    if survey == 2:
        return 1
    elif survey == 3:
        return 2
    elif survey == 4 or survey == 5:
        return 3
    return survey


def split_source(y_true,y_pred,source_info):
    mongodb_pred = []
    mongodb_target = []

    react_pred = []
    react_target = []

    socketio_pred = []
    socketio_target = []
    
    for p,t,s in zip(y_pred,y_true,source_info):
        if s[0] == 0:
            mongodb_pred.append(p)
            mongodb_target.append(t)
        elif s[0] == 1:
            react_pred.append(p)
            react_target.append(t)
        else:
            socketio_pred.append(p)
            socketio_target.append(t)
    mongodb = [mongodb_target, mongodb_pred]
    react = [react_target, react_pred]
    socketio = [socketio_target, socketio_pred]
    return {'mongodb':mongodb,
            'react':react,
            'socketio':socketio}