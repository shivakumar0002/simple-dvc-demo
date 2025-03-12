# Read the data from the data source
# Save it in the data/raw for further processing

import os
import argparse
from get_data import read_params, get_data


def load_and_save(config_path):
    """Reads configuration parameters, loads the dataset, and saves it in the raw data directory."""
    # Read parameters from config
    config = read_params(config_path)

    # Load dataset
    df = get_data(config_path)

    # Replace spaces in column names with underscores
    new_cols = [col.replace(" ", "_") for col in df.columns]

    # Fetch the save path from the config
    raw_data_path = config["load_data"]["raw_dataset_csv"]

    # Ensure the directory exists before saving
    raw_data_dir = os.path.dirname(raw_data_path)
    os.makedirs(raw_data_dir, exist_ok=True)  # Create the directory if it does not exist

    # Save the dataset
    df.to_csv(raw_data_path, sep=",", index=False, header=new_cols)
    print(f"✅ Data successfully saved at: {raw_data_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Load data from source and save it as raw dataset")
    parser.add_argument("--config", default="params.yaml", help="Path to configuration file")
    parsed_args = parser.parse_args()

    load_and_save(config_path=parsed_args.config)