from tqdm import tqdm
import os
import argparse
from preprocess.text_processing import text2sentence
from preprocess.utils import load_config, dynamic_import, append_and_save_csv
import sys
import nltk
from nltk.corpus import stopwords
nltk.download('stopwords')

def TextProcess():
    # Load configuration file
    config_path = os.path.join(root_dir, "preprocess", "feature_config.json")
    config = load_config(config_path)

    # Get transcript directory
    dir_path = os.path.join(root_dir, "data", "transcripts", f"{dataset}")
    if not os.path.exists(dir_path):
        print(f"Error: Directory {dir_path} does not exist.")
        return

    # Iterate through transcript files
    for file in tqdm(os.listdir(dir_path)):
        file_id = file.split(".")[0]
        # print(f"Processing File ID: {file_id}")

        # Process the transcript
        doc = text2sentence(file, dataset, root_dir)

        # Dynamically execute feature extraction modules
        for module_name, should_extract in config.items():
            if should_extract:
                module_name_relative="preprocess.extractors."+module_name
                module = dynamic_import(module_name_relative)  # Import module
                features = module.extract_features(doc)  # Extract features
                # print(features)
                output_file = os.path.join(root_dir, "features", f"{dataset}/{module_name}.csv")
                append_and_save_csv(output_file, features, file_id)  # Save features

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Extract textual features from transcripts.")
    parser.add_argument("--dataset", type=str, choices=["beginning", "end", "middle", "full"], required=True)
    args = parser.parse_args()

    dataset = args.dataset
    root_dir = os.path.abspath(os.path.dirname(__file__))
    root_dir = os.path.dirname(root_dir)  # Remove the last folder (e.g., preprocess)
    sys.path.append(root_dir)
    TextProcess()