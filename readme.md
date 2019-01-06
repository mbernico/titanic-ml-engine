## Titanic on GCP ML-Engine

Code me like one of your french girls...

This repo is a demo for using GCP ML-Engine to train a scikit learn model on GCP ML-Engine

## Setup
1. Install Python in a virtual environment and then install and configured the [GCP Cloud SDK](https://cloud.google.com/sdk/)
2. Install all python requirements `pip install -r requirements.txt`
3. Create a storage bucket using `create_bucket.sh`
4. Download train.csv and test.csv from [Kaggle's Titanic competition](https://www.kaggle.com/c/titanic) and place it in /data in this repo
5. Run data_prep.py

## data_prep.py
This script downloads the 'train.csv' file hosted in the GCP project bucket and then creates a training and validation dataset using that data. Finally the training and validation dataset are both uploaded back to the GCP project bucket

## Model Training

