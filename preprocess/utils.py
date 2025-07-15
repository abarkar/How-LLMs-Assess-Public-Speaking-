# utils.py
import os
import importlib
import json
import pandas as pd

def load_config(config_path):
    """Load the configuration file."""
    with open(config_path, "r") as f:
        return json.load(f)

def dynamic_import(module_name):
    """Dynamically import a module by name."""
    return importlib.import_module(module_name)

def append_and_save_csv(filepath, data, file_id):
    """
    Append new data to a CSV file and save it.
    
    Parameters:
        filepath (str): Path to the CSV file.
        data (dict): Dictionary of feature data to save.
        file_id (str): Identifier for the current file (saved in the 'category' column).
    """
    # Prepare the new row
    row = {"category": file_id, **data}
    columns = list(row.keys())

    # Check if the file already exists
    if os.path.exists(filepath):
        # Load the existing file
        df = pd.read_csv(filepath)

        # Ensure the columns match
        if set(df.columns) != set(columns):
            raise ValueError(f"Column mismatch in {filepath}. Expected columns: {df.columns}, Got: {columns}")
    else:
        # Create a new DataFrame if the file does not exist
        df = pd.DataFrame(columns=columns)

    # Append the new row
    new_row_df = pd.DataFrame([row])
    df = pd.concat([df, new_row_df], ignore_index=True)

    # Save the updated DataFrame to the CSV file
    df.to_csv(filepath, index=False)
