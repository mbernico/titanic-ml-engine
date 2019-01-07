# This script submits a simple training job to GCP.

#!/bin/sh

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