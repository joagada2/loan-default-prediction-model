import hydra
import pandas as pd
import numpy as np
from hydra.utils import to_absolute_path as abspath
from omegaconf import DictConfig
import os
import whylogs as why
from whylogs.api.writer.whylabs import WhyLabsWriter
import datetime
import joblib

#function to load training df
def get_data(raw_path: str):
    data = pd.read_csv(raw_path)
    return data

#function to drop feature
def drop_feat(df: pd.DataFrame, feat_to_drop: list):
    df = df.drop(columns=feat_to_drop, axis=1)
    return df

#function to load model
def load_model(model_path: str):
    return joblib.load(model_path)

#function to load processed data
def load_new_df(path: DictConfig):
    new_df_1 = pd.read_csv(abspath(path.new_df_1.path))
    new_df_2 = pd.read_csv(abspath(path.new_df_2.path))
    new_df_3 = pd.read_csv(abspath(path.new_df_3.path))
    new_df_4 = pd.read_csv(abspath(path.new_df_4.path))
    return new_df_1, new_df_2, new_df_3, new_df_4

#function to process data
def process_df(df):
    print(df.shape)
    # fill missing values in revol_util column
    df['revol_util'] = df['revol_util'].fillna(df['revol_util'].mean())

    #function for missing value handling on mort_acc column
    total_acc_avg = df.groupby('total_acc').mean()['mort_acc']
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

    df['mort_acc'] = df.apply(lambda x: fill_mort_acc(x['total_acc'], x['mort_acc']), axis=1)

    #function to fill missing values in pub_rec_bankruptcy column
    pub_rec_avg = df.groupby('pub_rec').mean()['pub_rec_bankruptcies']

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

    df['pub_rec_bankruptcies'] = df.apply(
        lambda x: fill_pub_rec_bankruptcies(x['pub_rec'], x['pub_rec_bankruptcies']), axis=1)

    # We simply take the numerical part and convert to integer
    df['term'] = df['term'].apply(lambda term: int(term[:3]))

    # group some values of the home ownership feature into "OTHER"
    df['home_ownership'] = df['home_ownership'].replace(['NONE', 'ANY'], 'OTHER')

    # extract year, convert to integer and use as earliest_cr_line
    df['earliest_cr_year'] = df['earliest_cr_line'].apply(lambda date: int(date[-4:]))

    # create zip_code feature by extracting zip code from address and remove address feature
    df['zip_code'] = df['address'].apply(lambda address: address[-5:])

    # convert the target feature to dummies
    df['loan_repaid'] = df['loan_status'].map({'Fully Paid': 1, 'Charged Off': 0})
    print(df.shape)
    return df

@hydra.main(version_base=None, config_path="../config", config_name="main")
def monitor_model(config: DictConfig):
    """Function to process the data"""

    #loand the training dataset
    data = get_data(abspath(config.processed.path))

    #split to features and target
    X = data.drop('loan_repaid', axis=1)
    y = data['loan_repaid']

    #load the monitoring datasets
    new_df_1, new_df_2, new_df_3, new_df_4 = load_new_df(config.raw_monitoring)

    #apply the process_df function
    new_df_1 = process_df(new_df_1)
    new_df_2 = process_df(new_df_2)
    new_df_3 = process_df(new_df_3)
    new_df_4 = process_df(new_df_4)

    print(new_df_1.shape)
    print(new_df_2.shape)
    print(new_df_3.shape)
    print(new_df_4.shape)

    # drop irrelevant features
    new_df_1 = drop_feat(new_df_1, config.feature.drop)
    new_df_2 = drop_feat(new_df_2, config.feature.drop)
    new_df_3 = drop_feat(new_df_3, config.feature.drop)
    new_df_4 = drop_feat(new_df_4, config.feature.drop)

    print(new_df_1.shape)
    print(new_df_2.shape)
    print(new_df_3.shape)
    print(new_df_4.shape)

    # convert the categorical variables to dummy variables
    new_df_1 = pd.get_dummies(new_df_1, drop_first=True)
    new_df_2 = pd.get_dummies(new_df_2, drop_first=True)
    new_df_3 = pd.get_dummies(new_df_3, drop_first=True)
    new_df_4 = pd.get_dummies(new_df_4, drop_first=True)

    print(new_df_1.shape)
    print(new_df_2.shape)
    print(new_df_3.shape)
    print(new_df_4.shape)

    # save processed data
    new_df_1.to_csv(abspath(config.processed_monitoring.new_df_1.path), index=False)
    new_df_2.to_csv(abspath(config.processed_monitoring.new_df_2.path), index=False)
    new_df_3.to_csv(abspath(config.processed_monitoring.new_df_3.path), index=False)
    new_df_4.to_csv(abspath(config.processed_monitoring.new_df_4.path), index=False)

    #split to features and targets
    X_new_df_1 = new_df_1.drop('loan_repaid',axis=1)
    X_new_df_2 = new_df_2.drop('loan_repaid',axis=1)
    X_new_df_3 = new_df_3.drop('loan_repaid',axis=1)
    X_new_df_4 = new_df_4.drop('loan_repaid',axis=1)

    y_new_df_1 = new_df_1['loan_repaid']
    y_new_df_2 = new_df_2['loan_repaid']
    y_new_df_3 = new_df_3['loan_repaid']
    y_new_df_4 = new_df_4['loan_repaid']

    print(X_new_df_1.shape)
    print(X_new_df_2.shape)
    print(X_new_df_3.shape)
    print(X_new_df_4.shape)

    print(y_new_df_1.shape)
    print(y_new_df_2.shape)
    print(y_new_df_3.shape)
    print(y_new_df_4.shape)

    #save the final monitoring dataset
    X_new_df_1.to_csv(abspath(config.final_monitoring.X_new_df_1.path), index=False)
    X_new_df_2.to_csv(abspath(config.final_monitoring.X_new_df_2.path), index=False)
    X_new_df_3.to_csv(abspath(config.final_monitoring.X_new_df_3.path), index=False)
    X_new_df_4.to_csv(abspath(config.final_monitoring.X_new_df_4.path), index=False)

    y_new_df_1.to_csv(abspath(config.final_monitoring.y_new_df_1.path), index=False)
    y_new_df_2.to_csv(abspath(config.final_monitoring.y_new_df_2.path), index=False)
    y_new_df_3.to_csv(abspath(config.final_monitoring.y_new_df_3.path), index=False)
    y_new_df_4.to_csv(abspath(config.final_monitoring.y_new_df_4.path), index=False)

    #create list of all the feature df
    df_X = [X_new_df_1,X_new_df_2,X_new_df_3,X_new_df_4]

    #create list of all target dataframe
    df_y = [y_new_df_1,y_new_df_2,y_new_df_3,y_new_df_4]

    # create profile
    profile1 = why.log(X_new_df_1)

    profile_view1 = profile1.view()
    profile_view1.to_pandas()
    print(profile_view1.to_pandas())

    # set authentication & project keys
    os.environ["WHYLABS_DEFAULT_ORG_ID"] = 'org-YNx5UG'
    os.environ["WHYLABS_API_KEY"] = 'YD0qo663PK.VJ5ZeKOjH14WMtHLGgMrvSLafZzaisZNyXQi054TRLaRmqJ5FB6J4'
    os.environ["WHYLABS_DEFAULT_DATASET_ID"] = 'model-4'

    # Single Profile
    writer = WhyLabsWriter()
    profile = why.log(X_new_df_1)
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
    model = load_model(abspath(config.model.path))

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

if __name__ == "__main__":
    monitor_model()