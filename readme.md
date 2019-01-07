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
A GCP training job is built as a python module. A GCP ready version of titanic is built in the cloud_train module in this repo.
To train on GCP ML-Engine run the script `submit_simple_training_job.sh`, but before you do, try submitting it locally with the script `submit_local_training_job.sh'.  Local training jobs are a great feature of GCP that allow the model builder debug the model before sending it away to train.

The submit script is really just a single (long) command specifying the runtime environmentals for the job.
```bash
PROJECT_ID=$(gcloud config list project --format "value(core.project)")
BUCKET_NAME=${PROJECT_ID}-mlengine
REGION="us-central1"
JOB_NAME=titanic_training_$(date +"%Y%m%d_%H%M%S")
JOB_DIR="gs://$BUCKET_NAME/titanic"
TRAINER_PACKAGE_PATH="./cloud_training/"
MAIN_TRAINER_MODULE="cloud_training.train"
RUNTIME_VERSION=1.12
PYTHON_VERSION=3.5
SCALE_TIER=BASIC

gcloud ml-engine jobs submit training $JOB_NAME \
  --job-dir $JOB_DIR \
  --package-path $TRAINER_PACKAGE_PATH \
  --module-name $MAIN_TRAINER_MODULE \
  --region $REGION \
  --runtime-version=$RUNTIME_VERSION \
  --python-version=$PYTHON_VERSION \
  --scale-tier $SCALE_TIER
```
When you submit the job GCP will provision a container that matches the [runtime-version you specify](https://cloud.google.com/ml-engine/docs/scikit/runtime-version-list).  If you need packages that aren't avaiable in that runtime version they [can be uploaded manually of via command line](https://cloud.google.com/ml-engine/docs/scikit/versioning)

## Local Inference
Just like local training jobs, you can test your model with local inference jobs as well.

### Things to watch out for...
[Bad Magic Number in pyc file](https://stackoverflow.com/questions/48824381/gcloud-ml-engine-local-predict-runtimeerror-bad-magic-number-in-pyc-file)

