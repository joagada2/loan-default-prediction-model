import pandas as pd
import numpy as np
import joblib
from datetime import datetime, timedelta
from prefect import Flow, Parameter, task
from prefect.schedules import IntervalSchedule

#task to load model
@task
def get_data():
    data = pd.read_csv('inferencing/input_data/input_df.csv')
    return data

#task to load model
@task
def load_model():
    model = joblib.load('model/loan_default_pred_model.pkl')
    return model

#task to process data
@task
def process_data(a):
    """Function to process the data"""
    # fill missing values in revol_util column
    a['revol_util'] = a['revol_util'].fillna(a['revol_util'].mean())

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

    # drop irrelevant features
    a = a.drop(['emp_title','emp_length','title','grade','issue_d','earliest_cr_line','address'], axis=1)

    print(a.shape)
    # convert the categorical variables to dummy variables
    a = pd.get_dummies(a, drop_first=True)

    # save processed data
    a.to_csv('inferencing/processed1/processed.csv', index=False)
    return a

#task to get and save predictions
@task
def predict_df(b,c):
    # get prediction
    prediction = c.predict(b)
    probability = c.predict_proba(b)
    b['predictions'] = prediction.tolist()
    b['prediction_probability'] = probability.tolist()
    #save prediction
    b.to_csv('inferencing/prediction/prediction_df.csv', index=False)

schedule = IntervalSchedule(interval=timedelta(hours=24))

with Flow('batch_inferencing_pipeline') as flow:
    data = get_data()
    model = load_model()
    processed_data = process_data(data)
    predict_df(processed_data, model)

flow.run()
#connect to prefect 1 cloud
flow.register(project_name='loan-default-prediction')
flow.run_agent()
