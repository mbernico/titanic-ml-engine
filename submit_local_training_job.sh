## This script submits a local training job.  Local jobs are useful for debugging.

#!/bin/sh

TRAINER_PACKAGE_PATH="./cloud_training/"
MAIN_TRAINER_MODULE="cloud_training.train"


gcloud ml-engine local train \
  --module-name $MAIN_TRAINER_MODULE \
  --package-path $TRAINER_PACKAGE_PATH \
