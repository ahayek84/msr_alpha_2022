# description of the file : 
"""
ALI
SVM and RFC Gridsearch from Sklearn 
Paramter tuning 
"""
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import classification_report
from preprocessors import create_dataset
import argparse


def get_params_model(model_type):
    if model_type == "random_forest":
        parameters = {"n_estimators": [50,75,100,150],
                        "max_depth": [4,5,6,7,8,10],
                        "max_features":["sqrt", "log2", "auto"],
                        "criterion" :["entropy"]
                        }
        model = RandomForestClassifier()
    else:
        parameters = {
            "C": [1, 10, 100, 1000],
            # "gamma": [1e-3, 1e-4],
            "max_iter": [500,100,1000,1500,2000,5000]
            }
        model = SVC(kernel="rbf", max_iter=2000)
    return parameters,model



def search_best_params(X_train,y_train,X_test,y_test,parameter,model):
    for score in ["precision_macro", "recall_macro","f1_macro"]:

        grid_search = GridSearchCV(model,
                                    parameter,
                                    n_jobs=-1,
                                    scoring = score)

        grid_search.fit(X_train, y_train)
        print("Grid search completed.")
        print(f"The best parameters for eval function {score} are: {grid_search.best_params_}")
        print(f"The best score for eval function {score} is: {grid_search.best_score_}")
        print(f"The complete classification report for eval function {score} is:")
        y_pred = grid_search.predict(X_test)
        print(classification_report(y_test, y_pred))


if __name__ == "__main__":
    parser = argparse.ArgumentParser("Fine-tuning for Random Forest and SVC using Grid search")
    parser.add_argument("--model_type", type=str, default="random_forest", help="Model type to fine-tune.")
    args = parser.parse_args()
    
    X_train, X_test, y_train, y_test = create_dataset()
    parameters, model = get_params_model(args.model_type)
    search_best_params(X_train,y_train,X_test,y_test,parameters, model)
