########################################################################
# data_prep.py                                                         #
# Split Kaggle Titanic's train.csv into a training and validation set  #
# suitable for use on Google's ML-Engine                               #
#                                                                      #
########################################################################
import pandas as pd
from google.cloud import storage
from sklearn.model_selection import train_test_split

def cache_titanic_data_from_cloud_bucket(BUCKET_NAME, cloud_filename, cache_name):
    bucket = storage.Client().bucket(BUCKET_NAME)
    blob = bucket.blob(cloud_filename)
    # Download the data
    blob.download_to_filename(cache_name)
    return

def load_cached_data(cache_name):
    with open(cache_name, 'r') as train_data:
        data = pd.read_csv(train_data)
    return data


def create_validation_set(df, test_size=0.2):
    train, val = train_test_split(df, test_size=0.2)
    return train, val

def save_data_to_cloud_bucket(df, BUCKET_NAME, cloud_filename, cache_name):
    bucket = storage.Client().bucket(BUCKET_NAME)
    df.to_csv(cache_name)
    blob = bucket.blob(cloud_filename)
    blob.upload_from_filename(cache_name)
    return

def main():
    BUCKET_NAME = 'sandbox-226501-mlengine'
    cloud_filename = 'titanic-train.csv'
    cache_name = './data/train.csv'
    cache_titanic_data_from_cloud_bucket(BUCKET_NAME, cloud_filename, cache_name)
    df = load_cached_data(cache_name)
    train, val = create_validation_set(df)
    save_data_to_cloud_bucket(train, BUCKET_NAME, cloud_filename="titanic_training_split.csv", cache_name="./data/titanic_train_temp.csv")
    save_data_to_cloud_bucket(val, BUCKET_NAME, cloud_filename="titanic_val_split.csv", cache_name="./data/titanic_val_temp.csv")


if __name__ == "__main__":
    main()

