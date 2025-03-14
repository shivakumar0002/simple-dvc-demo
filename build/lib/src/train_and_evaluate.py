# Load the train and test data
# Train the algorithm
# Save the metrics and parameters

import os
import warnings
import sys
import pandas as pd
import numpy as np
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.linear_model import ElasticNet
from get_data import read_params
import argparse
import joblib
import json


def eval_metrics(actual, pred):
    """
    Calculate evaluation metrics for the trained model.
    Returns RMSE, MAE, and R2 score.
    """
    rmse = np.sqrt(mean_squared_error(actual, pred))
    mae = mean_absolute_error(actual, pred)
    r2 = r2_score(actual, pred)
    return rmse, mae, r2


def train_and_evaluate(config_path):
    """
    Trains the ElasticNet model and evaluates it using the test data.
    Saves the trained model, evaluation metrics, and parameters.
    """
    # Read configuration parameters
    config = read_params(config_path)

    # Load dataset paths
    train_data_path = config["split_data"]["train_path"]
    test_data_path = config["split_data"]["test_path"]
    random_state = config["base"]["random_state"]
    model_dir = config["model_dir"]

    # Model hyperparameters (aligned with params.yaml)
    alpha = config["estimators"]["ElasticNet"]["params"]["alpha"]
    l1_ratio = config["estimators"]["ElasticNet"]["params"]["l1_ratio"]

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

    # Train the ElasticNet model
    lr = ElasticNet(alpha=alpha, l1_ratio=l1_ratio, random_state=random_state)
    lr.fit(train_x, train_y)

    # Make predictions
    predicted_qualities = lr.predict(test_x)

    # Evaluate model performance
    rmse, mae, r2 = eval_metrics(test_y, predicted_qualities)

    print(f"✅ ElasticNet model (alpha={alpha}, l1_ratio={l1_ratio}):")
    print(f"  RMSE: {rmse}")
    print(f"  MAE: {mae}")
    print(f"  R2: {r2}")

    # Save metrics (aligned with dvc.yaml)
    scores_file = config["reports"]["scores"]
    params_file = config["reports"]["params"]

    os.makedirs(os.path.dirname(scores_file), exist_ok=True)

    with open(scores_file, "w") as f:
        scores = {
            "rmse": rmse,
            "mae": mae,
            "r2": r2
        }
        json.dump(scores, f, indent=4)

    with open(params_file, "w") as f:
        params = {
            "alpha": alpha,
            "l1_ratio": l1_ratio,
        }
        json.dump(params, f, indent=4)

    # Save trained model (aligned with dvc.yaml)
    os.makedirs(model_dir, exist_ok=True)
    model_path = os.path.join(model_dir, "model.joblib")
    joblib.dump(lr, model_path)

    print(f"✅ Model saved at: {model_path}")
    print(f"✅ Metrics saved at: {scores_file}")
    print(f"✅ Params saved at: {params_file}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train and evaluate the model")
    parser.add_argument("--config", default="params.yaml", help="Path to configuration file")
    parsed_args = parser.parse_args()
    
    train_and_evaluate(config_path=parsed_args.config)