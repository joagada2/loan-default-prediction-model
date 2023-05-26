import hydra
import pandas as pd
import numpy as np
from hydra.utils import to_absolute_path as abspath
from omegaconf import DictConfig
from patsy import dmatrices
from sklearn.model_selection import train_test_split

def get_data(raw_path: str):
    data = pd.read_csv(raw_path)
    return data
def drop_feat(df:pd.DataFrame, feat_to_drop:list):
    df = df.drop(columns = feat_to_drop, axis=1)
    return df

@hydra.main(version_base=None, config_path="../../config", config_name="main")
def process_data(config: DictConfig):
    """Function to process the data"""

    #apply get data function to import dataset
    data = get_data(abspath(config.raw.path))

    #drop employment title
    #data = data.drop('emp_title',axis=1)

    #drop employment length
    #data = data.drop('emp_length',axis=1)

    #drop loan title
    #data = data.drop('title',axis=1)

    #fill missing values in revol_util column
    data['revol_util'] = data['revol_util'].fillna(data['revol_util'].mean())

    total_acc_avg = data.groupby('total_acc').mean()['mort_acc']
    
    def fill_mort_acc(total_acc,mort_acc):
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

    data['mort_acc'] = data.apply(lambda x: fill_mort_acc(x['total_acc'], x['mort_acc']), axis=1)

    pub_rec_avg = data.groupby('pub_rec').mean()['pub_rec_bankruptcies']
    
    def fill_pub_rec_bankruptcies(pub_rec,pub_rec_bankruptcies):
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
    
    data['pub_rec_bankruptcies'] = data.apply(lambda x: fill_pub_rec_bankruptcies(x['pub_rec'], x['pub_rec_bankruptcies']), axis=1)
    
    # We simply take the numerical part and convert to integer
    data['term'] = data['term'].apply(lambda term: int(term[:3]))

    #drop grade since grade is embeded in subgrade
    #data = data.drop('grade',axis=1)

    #group some values of the home ownership feature into "OTHER"
    data['home_ownership']=data['home_ownership'].replace(['NONE', 'ANY'], 'OTHER')

    #drop issue_d to avoid data leakage
    #data = data.drop('issue_d',axis=1)

    #extract year, convert to integer and use as earliest_cr_line
    data['earliest_cr_year'] = data['earliest_cr_line'].apply(lambda date:int(date[-4:]))
    #data = data.drop('earliest_cr_line',axis=1)

    #create zip_code feature by extracting zip code from address and remove address feature
    data['zip_code'] = data['address'].apply(lambda address:address[-5:])
    #data = data.drop('address',axis=1)

    #convert the target feature to dummies
    data['loan_repaid'] = data['loan_status'].map({'Fully Paid':1,'Charged Off':0})
    data = drop_feat(data, config.feature.drop)
    #no longer needed since it has been mapped to numerical values converted to loan_repaid 
    #data = data.drop('loan_status', axis = 1)
    print(data.shape)
    #convert the categorical variables to dummy variables
    data = pd.get_dummies(data,drop_first=True)

    # save processed data
    data.to_csv(abspath(config.processed.path),index=False)

    # get processed data
    data = get_data(abspath(config.processed.path))

    # split feature and targets
    X = data.drop('loan_repaid', axis=1)
    y = data['loan_repaid']

    X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=42)

    #Save final dataset data
    X_train.to_csv(abspath(config.final.X_train.path), index=False)
    X_test.to_csv(abspath(config.final.X_test.path), index=False)
    y_train.to_csv(abspath(config.final.y_train.path), index=False)
    y_test.to_csv(abspath(config.final.y_test.path), index=False)

if __name__ == "__main__":
    process_data()
