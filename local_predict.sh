#!/bin/sh
rm /Users/mike/google-cloud-sdk/lib/googlecloudsdk/command_lib/ml_engine/*.pyc

PROJECT_ID=$(gcloud config list project --format "value(core.project)")
BUCKET_NAME=${PROJECT_ID}-mlengine
REGION="us-central1"
MODEL_DIR="gs://$BUCKET_NAME/titanic_model"
FRAMEWORK="SCIKIT_LEARN"
INPUT_FILE="./data/test_survived.json"

gcloud ml-engine local predict --model-dir=$MODEL_DIR \
    --json-instances $INPUT_FILE \
    --framework $FRAMEWORK \
    --verbosity debug
