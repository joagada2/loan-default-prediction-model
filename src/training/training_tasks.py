from sklearn.model_selection import train_test_split
from datetime import datetime, timedelta
from prefect import Flow, Parameter, task
from prefect.schedules import IntervalSchedule
import warnings
warnings.filterwarnings(action="ignore")
import numpy as np
import xgboost as xgb
import joblib
import pandas as pd
from sklearn.metrics import balanced_accuracy_score, f1_score, \
    roc_auc_score, precision_score, recall_score
from xgboost import XGBClassifier
import os
import mlflow
from dagshub import DAGsHubLogger

@task
def get_data():
    data = pd.read_csv('data/raw/lending_club_loan_two.csv')
    return data

@task
def process_data(a):
    # fill missing values in revol_util column
    a['revol_util'] = a['revol_util'].fillna(a['revol_util'].mean())

    #fill missing values in mort_acc column using total_acc
    total_acc_avg = a.groupby('total_acc').mean()['mort_acc']

    def fill_mort_acc(total_acc, mort_acc):
        '''
        Accepts the total_acc and mort_acc values for the row.
        Checks if the mort_acc is NaN , if so, it returns the avg mort_acc value
        for the corresponding total_acc value for that row.

        total_acc_avg here should be a Series or dictionary containing the mapping of the
        groupby averages of mort_acc per total_acc values.
        '''
        if np.isnan(mort_acc):
            return total_acc_avg[total_acc]
        else:
            return mort_acc

    # fill missing values in pub_rec_bankruptcies using pub_re
    a['mort_acc'] = a.apply(lambda x: fill_mort_acc(x['total_acc'], x['mort_acc']), axis=1)
    pub_rec_avg = a.groupby('pub_rec').mean()['pub_rec_bankruptcies']

    def fill_pub_rec_bankruptcies(pub_rec, pub_rec_bankruptcies):
        '''
        Accepts the total_acc and mort_acc values for the row.
        Checks if the mort_acc is NaN , if so, it returns the avg mort_acc value
        for the corresponding total_acc value for that row.

        total_acc_avg here should be a Series or dictionary containing the mapping of the
        groupby averages of mort_acc per total_acc values.
        '''
        if np.isnan(pub_rec_bankruptcies):
            return pub_rec_avg[pub_rec]
        else:
            return pub_rec_bankruptcies

    a['pub_rec_bankruptcies'] = a.apply(
        lambda x: fill_pub_rec_bankruptcies(x['pub_rec'], x['pub_rec_bankruptcies']), axis=1)

    # We simply take the numerical part and convert to integer
    a['term'] = a['term'].apply(lambda term: int(term[:3]))

    # group some values of the home ownership feature into "OTHER"
    a['home_ownership'] = a['home_ownership'].replace(['NONE', 'ANY'], 'OTHER')

    # extract year, convert to integer and use as earliest_cr_line
    a['earliest_cr_year'] = a['earliest_cr_line'].apply(lambda date: int(date[-4:]))

    # create zip_code feature by extracting zip code from address and remove address feature
    a['zip_code'] = a['address'].apply(lambda address: address[-5:])
    # data = data.drop('address',axis=1)

    # convert the target feature to dummies
    a['loan_repaid'] = a['loan_status'].map({'Fully Paid': 1, 'Charged Off': 0})

    #drop irrelevant features
    a = a.drop(['emp_title','emp_length','title','grade','issue_d','earliest_cr_line','address','loan_status'],axis=1)

    print(a.shape)
    # convert the categorical variables to dummy variables
    a = pd.get_dummies(a, drop_first=True)

    # save processed data
    a.to_csv('data/processed/processed.csv', index=False)

    # get processed data
    a = pd.read_csv('data/processed/processed.csv')

    print(a.shape)

    # split feature and targets
    X = a.drop('loan_repaid', axis=1)
    y = a['loan_repaid']

    print(X.shape)
    print(y.shape)

    X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=42)

    # Save final dataset data
    X_train.to_csv('data/final/X_train.csv', index=False)
    X_test.to_csv('data/final/X_test.csv', index=False)
    y_train.to_csv('data/final/y_train.csv', index=False)
    y_test.to_csv('data/final/y_test.csv', index=False)
    return a

@task
def train_model(b):
    """Function to train the model"""
    # split feature and targets
    X = b.drop('loan_repaid', axis=1)
    y = b['loan_repaid']

    print(X.shape)
    print(y.shape)

    #split to training and test sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=42)

    evaluation = [(X_test, y_test)]

    #craete model instance with optimal parameter
    model = xgb.XGBClassifier(seed=42,
                              objective="binary:logistic",
                              gamma=0,
                              learning_rate=0.25,
                              max_depth=6,
                              reg_lambda=10,
                              scale_pos_weight=0.24,
                              subsample=0.9,
                              colsample_bytree=0.5,
                              use_label_encoder=False,
                              missing=0)
    #train model
    model.fit(X_train,
              y_train,
              verbose=True,
              early_stopping_rounds=10,
              eval_metric='aucpr',
              eval_set=evaluation)

    # Save model to model folder
    joblib.dump(model, 'model/loan_default_pred_model.pkl')

    # save model to app dir for deployment
    joblib.dump(model, 'app/model1/loan_default_pre_model_depl.pkl')

    return model

@task
def evaluate(c):
    #link up to dagshub MLFlow environment
    os.environ['MLFLOW_TRACKING_URI'] = 'https://dagshub.com/joe88data/loan-default-prediction-model.mlflow'
    os.environ['MLFLOW_TRACKING_USERNAME'] = 'joe88data'
    os.environ['MLFLOW_TRACKING_PASSWORD'] = 'e94114ca328c75772401898d749decb6dbcbeb21'
    with mlflow.start_run():
        # Load data and model
        X_test = pd.read_csv('data/final/X_test.csv')
        y_test = pd.read_csv('data/final/y_test.csv')

        # Get predictions
        prediction = c.predict(X_test)

        # Get metrics
        f1 = f1_score(y_test, prediction)
        print(f"F1 Score of this model is {f1}.")

        # get metrics
        accuracy = balanced_accuracy_score(y_test, prediction)
        print(f"Accuracy Score of this model is {accuracy}.")

        area_under_roc = roc_auc_score(y_test, prediction)
        print(f"Area Under ROC is {area_under_roc}.")

        precision = precision_score(y_test, prediction)
        print(f"Precision of this model is {precision}.")

        recall = recall_score(y_test, prediction)
        print(f"Recall for this model is {recall}.")

        # helper class for logging model and metrics
        class BaseLogger:
            def __init__(self):
                self.logger = DAGsHubLogger()

            def log_metrics(self, metrics: dict):
                mlflow.log_metrics(metrics)
                self.logger.log_metrics(metrics)

            def log_params(self, params: dict):
                mlflow.log_params(params)
                self.logger.log_hyperparams(params)
        logger = BaseLogger()
        # function to log parameters to dagshub and mlflow
        def log_params(c: XGBClassifier):
            logger.log_params({"model_class": type(c).__name__})
            model_params = c.get_params()

            for arg, value in model_params.items():
                logger.log_params({arg: value})

        # function to log metrics to dagshub and mlflow
        def log_metrics(**metrics: dict):
            logger.log_metrics(metrics)
        # log metrics to remote server (dagshub)
        log_params(c)
        log_metrics(f1_score=f1, accuracy_score=accuracy, area_Under_ROC=area_under_roc, precision=precision,
                recall=recall)
            # log metrics to local mlflow
            # mlflow.sklearn.log_model(model, "model")
            # mlflow.log_metric('f1_score', f1)
            # mlflow.log_metric('accuracy_score', accuracy)
            # mlflow.log_metric('area_under_roc', area_under_roc)
            # mlflow.log_metric('precision', precision)
            # mlflow.log_metric('recall', recall)

#adding schedule here automate the pipeline and make it run every 10 minutes
schedule = IntervalSchedule(interval=timedelta(minutes=10))

#create and run flow locally. To schedule to workflow to be automatically triggered every 4 hrs,
#add 'schedule' as Flow parameter (ie with Flow("loan-default-prediction", schedule)
with Flow("loan-default-prediction", schedule) as flow:
    data = get_data()
    processed_data = process_data(data)
    model = train_model(processed_data)
    evaluate(model)

#flow.visualize()
flow.run()
#connect to prefect 1 cloud
flow.register(project_name='loan-default-prediction')
flow.run_agent()