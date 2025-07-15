import os
import json
import csv

# Path to the folder containing JSON files
json_subfolder = 'llama3'
dimension = 'global'
criteria_group="storytelling"
criteria="metaphor"
version = '1'
test_number = '1'
model_name_with_test_number = f"{json_subfolder}-{test_number}"

# Path to the folder containing JSON files
json_parent_folder = os.path.join('terminaloutput', dimension, version, model_name_with_test_number)
# json_parent_folder = os.path.join('terminaloutput', criteria_group, criteria, version, model_name_with_test_number)

# Path to the folder where the CSV file will be saved
csv_folder = os.path.join('results', dimension, version, f"{json_subfolder}-{test_number}")
csv_filename = f'Annotation_of_{dimension}_prompt_version_{version}_test_number_{test_number}_model_{json_subfolder}_7b.csv'
# csv_folder = os.path.join('results', criteria_group, criteria, version, f"{json_subfolder}-{test_number}")
# csv_filename = f'Annotation_of_{criteria_group}_{criteria}_prompt_version_{version}_test_number_{test_number}_model_{json_subfolder}_7b.csv'

# Get a list of all JSON files in the folder
json_files = []

# Iterate through city sub-folders
for json_file in os.listdir(json_parent_folder):   
    if json_file.endswith(".json"):
        json_files.append({
            "transcript ID": os.path.splitext(json_file)[0],
            "path": os.path.join(json_parent_folder, json_file)
        })

# Check if there are any JSON files
if json_files:
    # Create the CSV file
    os.makedirs(csv_folder, exist_ok=True)
    csv_file_path = os.path.join(csv_folder, csv_filename)

    # Extract all unique fieldnames
    all_fieldnames = set()
    for json_data in json_files:
        with open(json_data["path"], 'r') as jsonfile:
            try:
                data = json.load(jsonfile)
                all_fieldnames.update(data.keys())
            except json.JSONDecodeError:
                print(f"Empty JSON file: {json_data['path']}")

    # Include additional fieldnames
    all_fieldnames.update(["transcript ID", "path"])

    # Open CSV file for writing
    with open(csv_file_path, 'w', newline='') as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=all_fieldnames, extrasaction='ignore')

        # Write header
        writer.writeheader()

        # Iterate through JSON files
        for json_data in json_files:
            # Load JSON data
            with open(json_data["path"], 'r') as jsonfile:
                try:
                    data = json.load(jsonfile)
                except json.JSONDecodeError:
                    print(f"Empty JSON file: {json_data['path']}")
                    data = {}  # Set empty data if JSON file is empty

            # Add transcript ID, category, city, and path to the data
            data.update(json_data)

            # Write data to CSV file
            writer.writerow(data)
else:
    print("No JSON files found.")
