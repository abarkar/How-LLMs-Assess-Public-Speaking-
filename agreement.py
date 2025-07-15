import pandas as pd
import os
from pingouin import intraclass_corr

# Variables
json_subfolder = 'llama3'
dimension = 'persuasiveness'
criteria_group = "storytelling"
criteria = "discours"
test_number = '1'
model_name_with_test_number = f"{json_subfolder}-{test_number}"

# Paths to the two versions of CSV files
version_4_csv_folder = os.path.join('results', criteria_group, criteria, '4', f"{json_subfolder}-{test_number}")
version_4_csv_filename = f'Annotation_of_{criteria_group}_{criteria}_prompt_version_4_test_number_{test_number}_model_{json_subfolder}_7b.csv'
version_4_file_path = os.path.join(version_4_csv_folder, version_4_csv_filename)

version_3_csv_folder = os.path.join('results', criteria_group, criteria, '3', f"{json_subfolder}-{test_number}")
version_3_csv_filename = f'Annotation_of_{criteria_group}_{criteria}_prompt_version_3_test_number_{test_number}_model_{json_subfolder}_7b.csv'
version_3_file_path = os.path.join(version_3_csv_folder, version_3_csv_filename)

# Load the CSV files
data_v4 = pd.read_csv(version_4_file_path)
data_v3 = pd.read_csv(version_3_file_path)

# Map ordinal ratings to numeric scores
ordinal_mapping = {"A": 3, "B": 2, "C": 1}

if "transcript ID" in data_v4.columns and "transcript ID" in data_v3.columns:
    merged_data = pd.merge(data_v4, data_v3, on="transcript ID", suffixes=('_v4', '_v3'))
    print(len(merged_data))

    # Extract transcript IDs and scores for both versions
    transcript_ids = merged_data["transcript ID"]
    scores_v4 = merged_data["score_v4"].map(ordinal_mapping)
    scores_v3 = merged_data["score_v3"].map(ordinal_mapping)

    # Prepare data for ICC computation
    icc_data = pd.DataFrame({
        "targets": list(transcript_ids) + list(transcript_ids),  # Duplicate transcript IDs for both raters
        "raters": ["rater1"] * len(scores_v4) + ["rater2"] * len(scores_v3),  # Assign raters
        "ratings": list(scores_v4) + list(scores_v3)  # Combine scores
    })
else:
    # If no ID column, align the rows directly (fallback)
    min_length = min(len(data_v4), len(data_v3))
    scores_v4 = data_v4["score"][:min_length].map(ordinal_mapping)
    scores_v3 = data_v3["score"][:min_length].map(ordinal_mapping)
    transcript_ids = range(min_length)  # Use indices as dummy targets if no transcript ID

    # Prepare data for ICC computation
    icc_data = pd.DataFrame({
        "targets": list(transcript_ids) + list(transcript_ids),
        "raters": ["rater1"] * min_length + ["rater2"] * min_length,
        "ratings": list(scores_v4) + list(scores_v3)
    })

# Check for missing or invalid data
icc_data = icc_data.dropna()  # Drop rows with NaN values resulting from unmapped categories

# Print the prepared DataFrame
print(icc_data)

# Compute ICC
icc_result = intraclass_corr(data=icc_data, targets="targets", raters="raters", ratings="ratings")

# Save ICC results to a CSV file
output_folder = './annotation_quality/agreement/'
os.makedirs(output_folder, exist_ok=True)
output_file_path = os.path.join(output_folder, f"{criteria}_ICC_scores.csv")

icc_result.to_csv(output_file_path, index=False)

print(f"ICC scores saved to {output_file_path}")
