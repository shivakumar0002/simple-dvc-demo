# Load the train and test data
# Train the algorithm
# Save the metrics, parameters, and track with MLflow

import os
import warnings
import sys
import pandas as pd
import numpy as np
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.linear_model import ElasticNet
from get_data import read_params
from urllib.parse import urlparse
import argparse
import joblib
import json
import mlflow
import mlflow.sklearn


def eval_metrics(actual, pred):
    """
    Compute evaluation metrics for model performance.
    Returns RMSE, MAE, and R2 score.
    """
    rmse = np.sqrt(mean_squared_error(actual, pred))
    mae = mean_absolute_error(actual, pred))
    r2 = r2_score(actual, pred)
    return rmse, mae, r2


def train_and_evaluate(config_path):
    """
    Trains the ElasticNet model, evaluates it, logs metrics to MLflow,
    and saves the trained model.
    """
    # Read configuration parameters
    config = read_params(config_path)

    # Load dataset paths
    train_data_path = config["split_data"]["train_path"]
    test_data_path = config["split_data"]["test_path"]
    random_state = config["base"]["random_state"]
    model_dir = config["model_dir"]

    # Model hyperparameters
    alpha = config["estimators"]["ElasticNet"]["alpha"]
    l1_ratio = config["estimators"]["ElasticNet"]["l1_ratio"]

    # Extract target column
    target = config["base"]["target_col"]

    # Load training and testing datasets
    train = pd.read_csv(train_data_path, sep=",")
    test = pd.read_csv(test_data_path, sep=",")

    # Debugging step: Print columns to verify target exists
    print("Columns in training dataset:", train.columns.tolist())

    if target not in train.columns:
        raise ValueError(f"❌ ERROR: Target column '{target}' not found in training dataset. Check params.yaml and train data!")

    if target not in test.columns:
        raise ValueError(f"❌ ERROR: Target column '{target}' not found in test dataset. Check params.yaml and test data!")

    # Split data into features (X) and target (y)
    train_y = train[target]
    test_y = test[target]
    train_x = train.drop(target, axis=1)
    test_x = test.drop(target, axis=1)

    ################### ✅ MLFLOW INTEGRATION ✅ ###############################
    mlflow_config = config["mlflow_config"]
    remote_server_uri = mlflow_config["remote_server_uri"]

    # Set MLflow tracking URI
    mlflow.set_tracking_uri(remote_server_uri)

    # Set MLflow experiment name
    mlflow.set_experiment(mlflow_config["experiment_name"])

    with mlflow.start_run(run_name=mlflow_config["run_name"]) as mlops_run:
        # Initialize and train model
        lr = ElasticNet(alpha=alpha, l1_ratio=l1_ratio, random_state=random_state)
        lr.fit(train_x, train_y)

        # Make predictions
        predicted_qualities = lr.predict(test_x)

        # Evaluate model performance
        rmse, mae, r2 = eval_metrics(test_y, predicted_qualities)

        # Log hyperparameters to MLflow
        mlflow.log_param("alpha", alpha)
        mlflow.log_param("l1_ratio", l1_ratio)

        # Log evaluation metrics to MLflow
        mlflow.log_metric("rmse", rmse)
        mlflow.log_metric("mae", mae)
        mlflow.log_metric("r2", r2)

        # Determine MLflow tracking storage type
        tracking_url_type_store = urlparse(mlflow.get_artifact_uri()).scheme

        # Log model to MLflow (Local or Remote Tracking)
        if tracking_url_type_store != "file":
            mlflow.sklearn.log_model(
                lr, 
                "model", 
                registered_model_name=mlflow_config["registered_model_name"]
            )
        else:
            mlflow.sklearn.save_model(lr, "model")

        print(f"✅ MLflow logging completed with experiment: {mlflow_config['experiment_name']}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train and evaluate the model with MLflow logging")
    parser.add_argument("--config", default="params.yaml", help="Path to configuration file")
    parsed_args = parser.parse_args()
    
    train_and_evaluate(config_path=parsed_args.config)