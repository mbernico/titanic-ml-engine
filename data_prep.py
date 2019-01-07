########################################################################
# data_prep.py                                                         #
# Split Kaggle Titanic's train.csv into a training and validation set  #
# suitable for use on Google's ML-Engine                               #
#                                                                      #
########################################################################
import pandas as pd
from google.cloud import storage
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestRegressor


def cache_titanic_data_from_cloud_bucket(BUCKET_NAME, cloud_filename, cache_name):
    bucket = storage.Client().bucket(BUCKET_NAME)
    blob = bucket.blob(cloud_filename)
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
    df.to_csv(cache_name, index=False)
    blob = bucket.blob(cloud_filename)
    blob.upload_from_filename(cache_name)
    return


def drop_columns(train, test):
    drop_cols = ['PassengerId', 'Name', 'Ticket', 'Cabin']
    train.drop(drop_cols, axis=1, inplace=True)
    test.drop(drop_cols, axis=1, inplace=True)
    return train, test


def binarize_sex(train, test):
    train['Sex'] = train['Sex'].map({'male': 1, 'female': 0, 1: 1, 0: 0})
    test['Sex'] = train['Sex'].map({'male': 1, 'female': 0, 1: 1, 0: 0})
    return train, test


def clean_embarked(train, test):
    train = pd.concat([train, pd.get_dummies(train['Embarked'], prefix='Embarked')], axis=1)
    train.drop('Embarked', axis=1, inplace=True)
    test = pd.concat([test, pd.get_dummies(test['Embarked'], prefix='Embarked')], axis=1)
    test.drop('Embarked', axis=1, inplace=True)
    return train, test


def impute_age(train, test):
    median_age = train['Age'].median()
    train['Age'].fillna(median_age, inplace=True)
    test['Age'].fillna(median_age, inplace=True)
    return train, test


def impute_fare(train, test):
    median_fare = train['Fare'].median()
    test['Fare'].fillna(median_fare, inplace=True)
    return train, test


def clean_data(train, test):
    train, test = drop_columns(train, test)
    train, test = binarize_sex(train, test)
    train, test = clean_embarked(train, test)
    train, test = impute_age(train, test)
    train, test = impute_fare(train, test)
    return train, test

def main():
    BUCKET_NAME = 'sandbox-226501-mlengine'
    cache_titanic_data_from_cloud_bucket(BUCKET_NAME, cloud_filename='titanic-train.csv', cache_name="./data/train.csv")
    cache_titanic_data_from_cloud_bucket(BUCKET_NAME, cloud_filename='titanic-test.csv', cache_name="./data/test.csv")
    df = load_cached_data(cache_name="./data/train.csv")
    test_df = load_cached_data(cache_name="./data/test.csv")
    df, test_df = clean_data(df, test_df)
    train, val = create_validation_set(df)
    save_data_to_cloud_bucket(train, BUCKET_NAME, cloud_filename="titanic_training_split.csv", cache_name="./data/titanic_train_temp.csv")
    save_data_to_cloud_bucket(val, BUCKET_NAME, cloud_filename="titanic_val_split.csv", cache_name="./data/titanic_val_temp.csv")
    save_data_to_cloud_bucket(test_df, BUCKET_NAME, cloud_filename="titanic_test_clean.csv", cache_name="./data/titanic_test_temp.csv")


if __name__ == "__main__":
    main()

