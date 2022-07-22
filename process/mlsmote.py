# -*- coding: utf-8 -*-
# Importing required Library
import numpy as np
import pandas as pd
import random
from sklearn.datasets import make_classification
from sklearn.neighbors import NearestNeighbors

def create_dataset(n_sample=1000):
    ''' 
    Create a unevenly distributed sample data set multilabel  
    classification using make_classification function
    
    args
    nsample: int, Number of sample to be created
    
    return
    X: pandas.DataFrame, feature vector dataframe with 10 features 
    y: pandas.DataFrame, target vector dataframe with 5 labels
    '''
    X, y = make_classification(n_classes=5, class_sep=2, 
                           weights=[0.1,0.025, 0.205, 0.008, 0.9], n_informative=3, n_redundant=1, flip_y=0,
                           n_features=10, n_clusters_per_class=1, n_samples=1000, random_state=10)
    #print(y)
    #y = pd.get_dummies(y, prefix='class')
    return pd.DataFrame(X), y

def MLSMOTE(X_train, y_train,synthetic_balance_proportion):
    import smote_variants as svs
    cols = X_train.columns
    oversampler = svs.SMOTE(proportion=synthetic_balance_proportion)
    oversampler = svs.MulticlassOversampling(oversampler)
    X_train, y_train = oversampler.sample(X_train.values, y_train)
    X_train_res = pd.DataFrame(columns=cols)
    i = 0
    for col in cols:
        X_train_res[col] = X_train[:,i] ##### llsls [:,2]
        i = i + 1
    return X_train_res, y_train

if __name__=='main':
    """
    main function to use the MLSMOTE
    """
    X, y = create_dataset()                     #Creating a Dataframe
    X_res,y_res =MLSMOTE(X, y, 0.12)     #Applying MLSMOTE to augment the dataframe
    print(X.shape,len(y),X_res.shape,len(y_res))
