#!/bin/sh +x

MODEL_NAME='titanic'
VERSION_NAME='v1'
INPUT_FILE='./data/test_survived.json'


gcloud ml-engine predict --model $MODEL_NAME  \
                   --version $VERSION_NAME \
                   --json-instances $INPUT_FILE
