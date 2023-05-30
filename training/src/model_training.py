import warnings
warnings.filterwarnings(action="ignore")
from functools import partial
from typing import Callable
import hydra
import joblib
import numpy as np
import pandas as pd
from hydra.utils import to_absolute_path as abspath
from hyperopt import STATUS_OK, Trials, fmin, hp, tpe
from omegaconf import DictConfig
from sklearn.metrics import accuracy_score
import xgboost as xgb

#function to load data
def load_data(path: DictConfig):
    X_train = pd.read_csv(abspath(path.X_train.path))
    X_test = pd.read_csv(abspath(path.X_test.path))
    y_train = pd.read_csv(abspath(path.y_train.path))
    y_test = pd.read_csv(abspath(path.y_test.path))
    return X_train, X_test, y_train, y_test

@hydra.main(version_base=None, config_path="../../config", config_name="main")
def train(config: DictConfig):
    """Function to train the model"""

    #load data
    X_train, X_test, y_train, y_test = load_data(config.final)

    evaluation = [(X_test, y_test)]

    model = xgb.XGBClassifier(seed=config.model.seed,
                                objective=config.model.objective,
                                gamma=config.model.gamma,
                                learning_rate=config.model.learning_rate,
                                max_depth=config.model.max_depth,
                                reg_lambda=config.model.reg_lambda,
                                scale_pos_weight=config.model.scale_pos_weight,
                                subsample=config.model.subsample,
                                colsample_bytree=config.model.colsample_bytree,
                                use_label_encoder=config.model.use_label_encoder,
                                missing=config.model.missing)

    model.fit(X_train,
            y_train,
            verbose=config.model.verbose,
            early_stopping_rounds=config.model.early_stopping_rounds,
            eval_metric=config.model.eval_metric,
            eval_set=evaluation)

    # Save model
    joblib.dump(model, abspath(config.model.path))
    #save model to app dir for deployment
    joblib.dump(model, abspath(config.model1.path))

if __name__ == "__main__":
    train()
