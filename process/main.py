# description of the file : 
"""
ALI
1- train the model based on best paramters from gridsearch 
2- Generate Precision - recall - Fscore matrix 
3- save the output in the data folder
"""
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.metrics import classification_report, roc_auc_score, cohen_kappa_score
from preprocessors import create_dataset
from utils import split_source
import argparse


def get_model(modle_type):
    if modle_type == "random_forest":
        model = RandomForestClassifier(criterion= 'entropy', max_depth=8,max_features= 'sqrt',n_estimators = 100)
    else:
        model = SVC(C= 100,max_iter = 1000)
    return model


def eval(y_true, y_pred):
    print(f'Classification report \n {classification_report(y_true, y_pred,zero_division=0)}')
    # print(f'AUC score {roc_auc_score(y_true,y_pred)}')
    print(f'Kappa score {cohen_kappa_score(y_true,y_pred)}')


def train_eval(data, model, all_source_eval=False):
    X_train, X_test, y_train, y_test, test_source_info = data
    model.fit(X_train,y_train)
    y_pred = model.predict(X_test)
    if all_source_eval:
        print("Get eval score for all the source type together")
        eval(y_test,y_pred)
    else:
        data_sources = split_source(y_test,y_pred,test_source_info.values)
        for k,v in data_sources.items():
            print(f"Get eval score for {k}")
            eval(v[0],v[1])


    

if __name__ == "__main__":
    parser = argparse.ArgumentParser("Train a Random Forest and SVC model")
    parser.add_argument("--model_type", type=str, default="random_forest", help="Model type to fine-tune.")
    parser.add_argument("--combine_labels", choices=["true","yes","false","no"],type=str.lower, help="Combine Novice and expert labels")
    args = parser.parse_args()
    
    combine_labels = True if args.combine_labels.lower() in ["true","yes"] else False
    data = create_dataset(use_smote=False,combine_labels=combine_labels)
    model = get_model(args.model_type)
    train_eval(data, model)