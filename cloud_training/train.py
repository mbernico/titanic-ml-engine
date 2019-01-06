from model import build_model
from util import configure_logging, save_model_to_cloud, cache_training_data, load_data
from sklearn.metrics import roc_auc_score

def main():
    BUCKET_NAME = 'sandbox-226501-mlengine'

    logger = configure_logging()
    logger.info("caching data from GCP bucket...")
    cache_training_data(BUCKET_NAME,
                        train_filename="titanic_training_split.csv",
                        val_filename="titanic_val_split.csv",
                        train_cache_name="titanic_train_temp.csv",
                        val_cache_name="titanic_val_temp.csv")

    data = load_data()
    logger.info("loaded data into dataframes")

    model = build_model()
    logger.info("loaded model")


    logger.info("beginning model fit")
    model.fit(data['X_train'], data['y_train'])
    logger.info("model has been fit")


    y_hat = model.predict_proba(data['X_val'])[:, 1]
    score = roc_auc_score(data['y_val'], y_hat)
    logger.info(" Model AUC:{}".format(score))
    save_model_to_cloud(model, BUCKET_NAME)
    logger.info("model saved to GCP")


if __name__ == "__main__":
    main()