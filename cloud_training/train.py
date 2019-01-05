import pandas as pd
import numpy as np
from google.cloud import storage
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score
import joblib
import logging
import sys
import datetime


class FareTransformer(TransformerMixin):

    def transform(self, fare, **transform_params):
        fare = fare.clip(0, np.percentile(fare, 99))
        return fare.astype('int')

    def fit(self, X, y=None, **fit_params):
        return self


def cache_titanic_data_from_cloud_bucket(BUCKET_NAME, cloud_filename, cache_name):
    bucket = storage.Client().bucket(BUCKET_NAME)
    blob = bucket.blob(cloud_filename)
    blob.download_to_filename(cache_name)
    return

def cache_training_data(BUCKET_NAME,train_filename, val_filename, train_cache_name, val_cache_name ):
    cache_titanic_data_from_cloud_bucket(BUCKET_NAME,
                                         cloud_filename=train_filename,
                                         cache_name=train_cache_name)

    cache_titanic_data_from_cloud_bucket(BUCKET_NAME,
                                         cloud_filename=val_filename,
                                         cache_name=val_cache_name)
    return


def load_data():
    train = pd.read_csv('../data/titanic_train_temp.csv')
    val = pd.read_csv('../data/titanic_val_temp.csv')
    return train, val


def build_model():
    sex_transformer = Pipeline([
        ('imputer', SimpleImputer(strategy='constant', fill_value='missing')),
        ('onehotencoder', OneHotEncoder())
    ])

    embarked_transformer = Pipeline([
        ('imputer', SimpleImputer(strategy='constant', fill_value='missing')),
        ('onehotencoder', OneHotEncoder())
    ])

    other_transformer = Pipeline([
        ('imputer', SimpleImputer(strategy='median'))
    ])

    age_transformer = Pipeline([
        ('imputer', SimpleImputer(strategy='median'))
        # we can add a custom model transformer later
    ])

    fare_transformer = Pipeline([
        ('imputer', SimpleImputer(strategy='median')),
        ('fare_transformer', FareTransformer())
    ])

    preprocessor = ColumnTransformer(
        transformers=[
            ('sex', sex_transformer, ['Sex']),
            ('age', age_transformer, ['Age']),
            ('embarked', embarked_transformer, ['Embarked']),
            ('other variables', other_transformer, ['Pclass', 'SibSp', 'Parch']),
            ('fare', fare_transformer, ['Fare']),
        ])

    model = Pipeline([
        ('preprocessor', preprocessor),
        ('randomForest', RandomForestClassifier(n_estimators=100))
    ])

    return model

def configure_logging():

    logger = logging.getLogger()
    logger.setLevel(logging.INFO)
    fmt = logging.Formatter("%(message)s")
    stdout_handler = logging.StreamHandler(sys.stdout)
    stdout_handler.setFormatter(fmt)
    logger.addHandler(stdout_handler)
    return logger


def save_model_to_cloud(model, BUCKET_NAME):
    model_name = 'model.joblib'
    joblib.dump(model, model_name)

    # Upload the model to GCS
    bucket = storage.Client().bucket(BUCKET_NAME)
    blob = bucket.blob('{}/{}'.format(
        datetime.datetime.now().strftime('titanic_%Y%m%d_%H%M%S'),
        model_name))
    blob.upload_from_filename(model_name)


def main():
    BUCKET_NAME = 'sandbox-226501-mlengine'

    logger = configure_logging()


    logger.info("caching data from GCP bucket...")
    cache_training_data(BUCKET_NAME,
                        train_filename="titanic_training_split.csv",
                        val_filename="titanic_val_split.csv",
                        train_cache_name="titanic_train_temp.csv",
                        val_cache_name="titanic_val_temp.csv")

    train, val = load_data()
    logger.info("loaded data into dataframes")
    model = build_model()
    logger.info("loaded model")
    X = train.drop('Survived', axis=1)
    y = train['Survived']
    logger.info("beginning model fit")
    model.fit(X, y)
    logger.info("model has been fit")

    X_val = val.drop('Survived', axis=1)
    y_val = val['Survived']
    y_hat = model.predict_proba(X_val)[:, 1]
    score = roc_auc_score(y_val, y_hat)
    logger.info(" Model AUC:{}".format(score))
    save_model_to_cloud(model, BUCKET_NAME)
    logger.info("model saved to GCP")


if __name__ == "__main__":
    main()