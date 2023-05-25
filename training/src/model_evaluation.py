import warnings

warnings.filterwarnings(action="ignore")

import hydra
import joblib
import mlflow
import pandas as pd
from helper import BaseLogger
from hydra.utils import to_absolute_path as abspath
from omegaconf import DictConfig
from sklearn.metrics import balanced_accuracy_score, f1_score, roc_auc_score, precision_score, recall_score
from xgboost import XGBClassifier
import os
#import mlflow.sklearn

logger = BaseLogger()

#function to load data
def load_data(path: DictConfig):
    X_test = pd.read_csv(abspath(path.X_test.path))
    y_test = pd.read_csv(abspath(path.y_test.path))
    return X_test, y_test

#function to load model
def load_model(model_path: str):
    return joblib.load(model_path)

#function to get prediction
def predict(model: XGBClassifier, X_test: pd.DataFrame):
    return model.predict(X_test)

#function to log parameters to dagshub and mlflow
def log_params(model: XGBClassifier):
    logger.log_params({"model_class": type(model).__name__})
    model_params = model.get_params()

    for arg, value in model_params.items():
        logger.log_params({arg: value})

#function to log metrics to dagshub and mlflow
def log_metrics(**metrics: dict):
    logger.log_metrics(metrics)

@hydra.main(version_base=None, config_path="../../config", config_name="main")
def evaluate(config: DictConfig):
    #mlflow.set_experiment('Customer_Churn')
    os.environ['MLFLOW_TRACKING_URI']=config.mlflow_tracking_ui
    os.environ['MLFLOW_TRACKING_USERNAME']=config.mlflow_USERNAME
    os.environ['MLFLOW_TRACKING_PASSWORD']=config.mlflow_PASSWORD

    with mlflow.start_run():

        # Load data and model
        X_test, y_test = load_data(config.final)
        model = load_model(abspath(config.model.path))

        # Get predictions
        prediction = predict(model, X_test)

        # Get metrics
        f1 = f1_score(y_test, prediction)
        print(f"F1 Score of this model is {f1}.")

        #get metrics
        accuracy = balanced_accuracy_score(y_test, prediction)
        print(f"Accuracy Score of this model is {accuracy}.")

        area_under_roc = roc_auc_score(y_test,prediction)
        print(f"Area Under ROC is {area_under_roc}.")

        precision = precision_score(y_test, prediction)
        print(f"Precision of this model is {precision}.")

        recall = recall_score(y_test, prediction)
        print(f"Recall for this model is {recall}.")

        #log metrics to remote server (dagshub)
        log_params(model)
        log_metrics(f1_score=f1, accuracy_score=accuracy, area_Under_ROC = area_under_roc, precision = precision, recall = recall)

        #log metrics to local mlflow
        #mlflow.sklearn.log_model(model, "model")
        #mlflow.log_metric('f1_score', f1)
        #mlflow.log_metric('accuracy_score', accuracy)
        #mlflow.log_metric('area_under_roc', area_under_roc)
        #mlflow.log_metric('precision', precision)
        #mlflow.log_metric('recall', recall)

if __name__ == "__main__":
    evaluate()
