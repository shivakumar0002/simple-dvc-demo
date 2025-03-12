# Load the train and test data
# Train the algorithm
# Save the metrics and parameters

import os
import argparse
import joblib
import json
import pandas as pd
import numpy as np
from sklearn.linear_model import ElasticNet
from sklearn.metrics import mean_squared_error, r2_score
from get_data import read_params

def train_and_evaluate(config_path):
    """
    Trains the model and evaluates it using the testing data.
    Saves the trained model and evaluation metrics.
    """
    # Read configuration parameters
    config = read_params(config_path)
    
    # Load training and testing datasets
    train_data_path = config["split_data"]["train_path"]
    test_data_path = config["split_data"]["test_path"]
    train = pd.read_csv(train_data_path, sep=",", encoding="utf-8")
    test = pd.read_csv(test_data_path, sep=",", encoding="utf-8")
    
    # Extract target column
    target = config["base"]["target_col"]
    train_y = train[target]
    test_y = test[target]
    train_x = train.drop(target, axis=1)
    test_x = test.drop(target, axis=1)
    
    # Train the model
    model = ElasticNet(
        alpha=config["estimators"]["ElasticNet"]["alpha"],
        l1_ratio=config["estimators"]["ElasticNet"]["l1_ratio"],
        random_state=config["base"]["random_state"]
    )
    model.fit(train_x, train_y)
    
    # Make predictions
    predictions = model.predict(test_x)
    
    # Evaluate the model
    mse = mean_squared_error(test_y, predictions)
    rmse = np.sqrt(mse)
    r2 = r2_score(test_y, predictions)
    
    # Save the model
    model_dir = config["model_dir"]
    os.makedirs(model_dir, exist_ok=True)
    model_path = os.path.join(model_dir, "model.joblib")
    joblib.dump(model, model_path)
    
    # Save evaluation metrics
    scores = {
        "rmse": rmse,
        "r2": r2
    }
    scores_path = config["reports"]["scores"]
    os.makedirs(os.path.dirname(scores_path), exist_ok=True)
    with open(scores_path, "w") as f:
        json.dump(scores, f, indent=4)
    
    # Save model parameters
    params = {
        "alpha": config["estimators"]["ElasticNet"]["alpha"],
        "l1_ratio": config["estimators"]["ElasticNet"]["l1_ratio"]
    }
    params_path = config["reports"]["params"]
    with open(params_path, "w") as f:
        json.dump(params, f, indent=4)
    
    print(f"✅ Model saved at: {model_path}")
    print(f"✅ Evaluation metrics saved at: {scores_path}")
    print(f"✅ Model parameters saved at: {params_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train and evaluate the model")
    parser.add_argument("--config", default="params.yaml", help="Path to configuration file")
    parsed_args = parser.parse_args()
    
    train_and_evaluate(config_path=parsed_args.config)