# -*- coding: utf-8 -*-
# Importing required Library
import pandas as pd


def MLSMOTE(X_train, y_train,synthetic_balance_proportion=0.3,n_neighbors=3):
    """
    Smote using the parameter given by author
    """
    import smote_variants as svs
    cols = X_train.columns
    oversampler = svs.SMOTE(random_state=0 , proportion=synthetic_balance_proportion,n_neighbors=n_neighbors)
    oversampler = svs.MulticlassOversampling(oversampler)
    
    X_train, y_train = oversampler.sample(X_train, y_train)
    X_train_res = pd.DataFrame(columns=cols)
    i = 0
    for col in cols:
        X_train_res[col] = X_train[:,i] ##### llsls [:,2]
        i = i + 1
    return X_train_res, y_train
