import pandas as pd
import spacy
import re
import os

# Load the French language model
nlp = spacy.load("fr_core_news_sm")

# Function to extract grade (A, B, C) and explanation from text
def extract_note_and_explanation(text):
    # note_match = re.search(r"Note\s*:\s*(A|B|C)", text)  # Match the Note
    # Regex to match both "Note : <NUM>" and "<NUM> : sometext" where <NUM> is a digit (e.g., 1, 2, 3)
    note_match = re.search(r"(Note\s*:\s*(\d+))|((\d+)\s*:\s*\w+)", text)
    
    # Match the Explanation (following 'Explication :')
    explanation_match = re.search(r"Explication\s*:\s*(.*)", text, re.DOTALL) 
    print(note_match)
    print(explanation_match)
    # Extract the note if matched
    note = note_match.group(1) if note_match else None
    
    # Extract the explanation if matched
    explanation = explanation_match.group(1) if explanation_match else None
    
    return note, explanation




# Path setup (unchanged)
json_subfolder = 'llama3'
dimension = 'persuasiveness'
criteria_group = "storytelling"
criteria = "metaphor"
version = '2'
test_number = '1'
model_name_with_test_number = f"{json_subfolder}-{test_number}"

# Path to the folder where the CSV file will be saved
# csv_folder = os.path.join('results', criteria_group, criteria, version, f"{json_subfolder}-{test_number}")
# csv_filename = f'Annotation_of_{criteria_group}_{criteria}_prompt_version_{version}_test_number_{test_number}_model_{json_subfolder}_7b.csv'
csv_folder = os.path.join('results', dimension, version, f"{json_subfolder}-{test_number}")
csv_filename = f'Annotation_of_{dimension}_prompt_version_{version}_test_number_{test_number}_model_{json_subfolder}_7b.csv'
answers_file_path = os.path.join(csv_folder, csv_filename)

# Read the CSV file
answers_df = pd.read_csv(answers_file_path, header=0)

# Apply the function to extract Note and Explication, and create new columns
answers_df[["score", "explanations"]] = answers_df["response"].apply(
    lambda text: pd.Series(extract_note_and_explanation(text))
)

# Function to check the quality of the answer based on the extracted grade
def check_quality(grade):
    if grade in ['A', 'B', 'C']:
        return 1  # Since grade is valid, mark it as good quality
    return 0

# Apply the function to determine the quality of the answer and create the "quality of answer" column
answers_df["quality of answer"] = answers_df["score"].apply(check_quality)

# Save the updated DataFrame back to the CSV file
answers_df.to_csv(answers_file_path, index=False)

print("Extraction and saving completed.")
