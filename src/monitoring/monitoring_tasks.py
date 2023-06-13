from datetime import datetime, timedelta
from prefect import Flow, Parameter, task
from prefect.schedules import IntervalSchedule
import pandas as pd
import numpy as np
import whylogs as why
from whylogs.api.writer.whylabs import WhyLabsWriter
import datetime
import joblib
import os

#task to load dataframes 1 to 4
@task
def load_first_df():
    new_df_1 = pd.read_csv('data/monitoring_data/raw_monitoring/new_df_1.csv')
    return new_df_1

@task
def load_second_df():
    new_df_2 = pd.read_csv('data/monitoring_data/raw_monitoring/new_df_2.csv')
    return new_df_2

@task
def load_third_df():
    new_df_3 = pd.read_csv('data/monitoring_data/raw_monitoring/new_df_3.csv')
    return new_df_3

@task
def load_fourth_df():
    new_df_4 = pd.read_csv('data/monitoring_data/raw_monitoring/new_df_4.csv')
    return new_df_4

@task
def get_training_data():
    training_data = pd.read_csv('data/raw/lending_club_loan_two.csv')
    return training_data

#function to process data
@task
def process_df(a):
    print(a.shape)
    # fill missing values in revol_util column
    a['revol_util'] = a['revol_util'].fillna(a['revol_util'].mean())

    #function for missing value handling on mort_acc column
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

    #function to fill missing values in pub_rec_bankruptcy column
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

    # convert the target feature to dummies
    a['loan_repaid'] = a['loan_status'].map({'Fully Paid': 1, 'Charged Off': 0})
    print(a.shape)
    return a

#tasks for further processing
@task
def drop_feature(b):
    new_df_drop = b.drop(['emp_title','emp_length','title','grade','issue_d','earliest_cr_line','address','loan_status'],axis=1)
    return new_df_drop

@task
def get_dummies(c):
    new_df_dummies = pd.get_dummies(c, drop_first=True)
    return new_df_dummies

@task
def save_processed_df(d):
    d.to_csv('data/monitoring_data/processed_monitoring/d.csv')

@task
def split_data_x(e):
    X_new_df = e.drop('loan_repaid',axis=1)
    return X_new_df

@task
def split_data_y(f):
    y_new_df = f['loan_repaid']
    return y_new_df

@task
def save_final_df(g):
    g.to_csv('data/monitoring_data/g.csv')

@task
def monitor_model(h,h1,h2,h3,h4,i1,i2,i3,i4):
    #split to features and target
    X = h.drop('loan_repaid', axis=1)
    y = h['loan_repaid']
    #create list of all the feature df
    df_X = [h1,h2,h3,h4]

    #create list of all target dataframe
    df_y = [i1,i2,i3,i4]

    # create profile
    profile1 = why.log(h1)

    profile_view1 = profile1.view()
    profile_view1.to_pandas()
    print(profile_view1.to_pandas())

    # set authentication & project keys
    os.environ["WHYLABS_DEFAULT_ORG_ID"] = 'org-YNx5UG'
    os.environ["WHYLABS_API_KEY"] = 'YD0qo663PK.VJ5ZeKOjH14WMtHLGgMrvSLafZzaisZNyXQi054TRLaRmqJ5FB6J4'
    os.environ["WHYLABS_DEFAULT_DATASET_ID"] = 'model-5'

    # Single Profile
    writer = WhyLabsWriter()
    profile = why.log(h1)
    writer.write(file=profile.view())

    # back fill 1 day per batch
    writer = WhyLabsWriter()
    for i, df in enumerate(df_X):
        # walking backwards. Each dataset has to map to a date to show up as a different batch in WhyLabs
        dt = datetime.datetime.now(tz=datetime.timezone.utc) - datetime.timedelta(days=i)

        # create profile for each batch of data
        profile = why.log(df).profile()

        # set the dataset timestamp for the profile
        profile.set_dataset_timestamp(dt)
        # write the profile to the WhyLabs platform
        writer.write(file=profile.view())

    #reference profile
    ref_profile = why.log(X).profile()
    writer = WhyLabsWriter().option(reference_profile_name="training_data_profile")
    writer.write(file=ref_profile.view())

    #Logging output
    pred_df_X = df_X
    model = joblib.load('model/loan_default_pred_model.pkl')

    for i, df in enumerate(pred_df_X):
        y_pred = model.predict(df)
        y_prob = model.predict_proba(df)
        pred_scores = []
        pred_classes = []

        for pred in y_pred:
            pred_classes.append(pred)
        df['class_output'] = pred_classes
        for prob in y_prob:
            pred_scores.append(max(prob))
        df['prob_output'] = pred_scores
        print(pred_scores)

    writer = WhyLabsWriter()
    for i, df in enumerate(pred_df_X):
        out_df = df[['class_output', 'prob_output']].copy()
        # walking backwards. Each dataset has to map to a date to show up as a different batch in WhyLabs
        dt = datetime.datetime.now(tz=datetime.timezone.utc) - datetime.timedelta(days=i)
        profile = why.log(out_df).profile()

        # set the dataset timestamp for the profile
        profile.set_dataset_timestamp(dt)
        # write the profile to the WhyLabs platform
        writer.write(file=profile.view())

    # Append ground truth data to dataframe
    for i, df in enumerate(pred_df_X):
        df['ground_truth'] = df_y[i]

    # Log performance
    #print(pred_df_X[0])
    for i, df in enumerate(pred_df_X):
        results = why.log_classification_metrics(
            df,
            target_column="ground_truth",
            prediction_column="class_output",
            score_column="prob_output"
        )
        # walking backwards. Each dataset has to map to a date to show up as a different batch in WhyLabs
        dt = datetime.datetime.now(tz=datetime.timezone.utc) - datetime.timedelta(days=i)

        profile = results.profile()
        profile.set_dataset_timestamp(dt)

        results.writer("whylabs").write()

schedule = IntervalSchedule(interval=timedelta(hours=24))

with Flow('model_monitoring_pipeline',schedule) as flow:
    new_df_1 = load_first_df()
    new_df_2 = load_second_df()
    new_df_3 = load_third_df()
    new_df_4 = load_fourth_df()
    training_data = get_training_data()

    processed_new_df_1 = process_df(new_df_1)
    processed_new_df_2 = process_df(new_df_2)
    processed_new_df_3 = process_df(new_df_3)
    processed_new_df_4 = process_df(new_df_4)
    training_data = process_df(training_data)

    new_df_1_drop = drop_feature(processed_new_df_1)
    new_df_2_drop = drop_feature(processed_new_df_2)
    new_df_3_drop = drop_feature(processed_new_df_3)
    new_df_4_drop = drop_feature(processed_new_df_4)
    training_data = drop_feature(training_data)

    new_df_dum_1 = get_dummies(new_df_1_drop)
    new_df_dum_2 = get_dummies(new_df_2_drop)
    new_df_dum_3 = get_dummies(new_df_3_drop)
    new_df_dum_4 = get_dummies(new_df_4_drop)
    training_data = get_dummies(training_data)

    save_processed_df(new_df_dum_1)
    save_processed_df(new_df_dum_2)
    save_processed_df(new_df_dum_3)
    save_processed_df(new_df_dum_4)

    X_new_df_1 = split_data_x(new_df_dum_1)
    X_new_df_2 = split_data_x(new_df_dum_2)
    X_new_df_3 = split_data_x(new_df_dum_3)
    X_new_df_4 = split_data_x(new_df_dum_4)

    y_new_df_1 = split_data_y(new_df_dum_1)
    y_new_df_2 = split_data_y(new_df_dum_2)
    y_new_df_3 = split_data_y(new_df_dum_3)
    y_new_df_4 = split_data_y(new_df_dum_4)

    save_final_df(X_new_df_1)
    save_final_df(X_new_df_2)
    save_final_df(X_new_df_3)
    save_final_df(X_new_df_4)

    save_final_df(y_new_df_1)
    save_final_df(y_new_df_2)
    save_final_df(y_new_df_3)
    save_final_df(y_new_df_4)

    monitor_model(training_data, X_new_df_1,X_new_df_2,X_new_df_3,X_new_df_4,
                  y_new_df_1,y_new_df_2,y_new_df_3,y_new_df_4)

flow.run()
#connect to prefect 1 cloud
flow.register(project_name='loan-default-prediction')
flow.run_agent()