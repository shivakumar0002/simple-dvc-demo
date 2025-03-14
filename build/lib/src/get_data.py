# Read parameters
# Process data
# Return dataframe

import os
import yaml
import pandas as pd
import argparse

def read_params(config_path):
    """Reads the YAML configuration file and returns its contents as a dictionary."""
    with open(config_path, "r") as yaml_file:
        config = yaml.safe_load(yaml_file)
    return config

def get_data(config_path):
    """Reads configuration parameters and loads the dataset."""
    config = read_params(config_path)
    
    # Fetch data path from config
    data_path = config["data_source"]["s3_source"]

    # Load dataset
    data = pd.read_csv(data_path, sep=",", encoding="utf-8")  # Specify encoding format
    return data  # Return dataframe

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Process and load dataset based on config file")
    parser.add_argument("--config", default="params.yaml", help="Path to configuration file")
    parsed_args = parser.parse_args()

    data = get_data(config_path=parsed_args.config)
    print(data.head())  # Print the first few rows of the dataframe to verify