MODEL_NAME="titanic"


gcloud ml-engine models create MODEL_NAME


gcloud ml-engine versions create v1 \
                          --model MODEL_NAME \
                          --origin gs://sandbox-226501-mlengine/titanic_model \
                          --framework "SCIKIT-LEARN" \
                          --runtime-version=1.12 \
                          --python-version=3.5

