# Split the raw data
# Save it in data/processed folder

import os
import argparse
import pandas as pd
from sklearn.model_selection import train_test_split
from get_data import read_params

def split_and_save_data(config_path):
    """
    Splits the raw data into training and testing sets and saves them.
    """
    # Read configuration parameters
    config = read_params(config_path)
    
    # Load raw dataset
    raw_data_path = config["load_data"]["raw_dataset_csv"]
    df = pd.read_csv(raw_data_path, sep=",", encoding="utf-8")
    
    # Split data into training and testing sets
    train, test = train_test_split(
        df, 
        test_size=config["split_data"]["test_size"], 
        random_state=config["base"]["random_state"]
    )
    
    # Fetch save paths from config
    train_data_path = config["split_data"]["train_path"]
    test_data_path = config["split_data"]["test_path"]
    
    # Ensure the directories exist before saving
    os.makedirs(os.path.dirname(train_data_path), exist_ok=True)
    os.makedirs(os.path.dirname(test_data_path), exist_ok=True)
    
    # Save the datasets
    train.to_csv(train_data_path, sep=",", index=False, encoding="utf-8")
    test.to_csv(test_data_path, sep=",", index=False, encoding="utf-8")
    print(f" Training data saved at: {train_data_path}")
    print(f" Testing data saved at: {test_data_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Split raw data into training and testing sets")
    parser.add_argument("--config", default="params.yaml", help="Path to configuration file")
    parsed_args = parser.parse_args()
    
    split_and_save_data(config_path=parsed_args.config)