import pickle
import logging
from google.cloud import storage
import datetime
import sys
import pandas as pd


def configure_logging():

    logger = logging.getLogger()
    logger.setLevel(logging.INFO)
    fmt = logging.Formatter("%(message)s")
    stdout_handler = logging.StreamHandler(sys.stdout)
    stdout_handler.setFormatter(fmt)
    logger.addHandler(stdout_handler)
    return logger


def save_model_to_cloud(model, BUCKET_NAME):
    model_name = 'model.pkl'
    with open(model_name,'wb')as handle:
        pickle.dump(model, handle)

    # Upload the model to GCS
    bucket = storage.Client().bucket(BUCKET_NAME)
    blob = bucket.blob('titanic_model/{}'.format(model_name))
    blob.upload_from_filename(model_name)


def cache_data_from_cloud_bucket(BUCKET_NAME, cloud_filename, cache_name):
    bucket = storage.Client().bucket(BUCKET_NAME)
    blob = bucket.blob(cloud_filename)
    blob.download_to_filename(cache_name)
    return

def cache_training_data(BUCKET_NAME,train_filename, val_filename, train_cache_name, val_cache_name ):
    cache_data_from_cloud_bucket(BUCKET_NAME,
                                         cloud_filename=train_filename,
                                         cache_name=train_cache_name)

    cache_data_from_cloud_bucket(BUCKET_NAME,
                                         cloud_filename=val_filename,
                                         cache_name=val_cache_name)
    return


def load_data():
    train = pd.read_csv('titanic_train_temp.csv')
    val = pd.read_csv('titanic_val_temp.csv')
    data={}
    data['X_train'] = train.drop('Survived', axis=1)
    data['y_train'] = train['Survived']
    data['X_val'] = val.drop('Survived', axis=1)
    data['y_val'] = val['Survived']
    return data



