import pandas as pd
import spacy
import re
import os

# # Load the English language model
# nlp = spacy.load("en_core_web_sm")

# # Function to extract grade (A, B, C) from text
# def extract_grade(text):
#     match = re.search(r"Grade : \s*(A|B|C)", text)  # Use regex to find the grade pattern
#     if match:
#         return match.group(1)  # Return the matched grade (A, B, or C)
#     else:
#         return None
# Load the French language model
nlp = spacy.load("fr_core_news_sm")

# Function to extract grade (A, B, C) from text in different formats
def extract_grade(text):
    # Regex to match both "Note : <MASK>" and "<MASK> : sometext"
    # print(text)
    match = re.search(r"(Note\s*:\s*(A|B|C))|((A|B|C)\s*:\s*\w+)", text)
    
    if match:
        # Check which group matched and return the correct grade
        if match.group(2):  # For "Note : A", "Note : B", or "Note : C"
            return match.group(2)
        elif match.group(4):  # For "A : sometext", "B : sometext", or "C : sometext"
            return match.group(4)
    else:
        return None

# Function to extract numerical grades from text in different formats
def extract_numerical_grade(text):
    # Regex to match both "Note : <NUM>" and "<NUM> : sometext" where <NUM> is a digit (e.g., 1, 2, 3)
    match = re.search(r"(Note\s*:\s*(\d+))|((\d+)\s*:\s*\w+)", text)
    
    if match:
        # Check which group matched and return the correct numerical grade
        if match.group(2):  # For "Note : 1", "Note : 2", etc.
            return int(match.group(2))
        elif match.group(4):  # For "1 : sometext", "2 : sometext", etc.
            return int(match.group(4))
    else:
        return None


# Function to extract numerical entities from text
def extract_numerical_entities(text):
    doc = nlp(text)
    numerical_entities = []
    for ent in doc.ents:
        if ent.label_ == "CARDINAL":  # Check if the entity is a cardinal number
            numerical_entities.append(ent.text)
    return numerical_entities

# Load the CSV files
json_subfolder = 'llama3'
dimension = 'global'
criteria_group="storytelling"
criteria="metaphor"
version = '1'
test_number = '1'
model_name_with_test_number = f"{json_subfolder}-{test_number}"

# Path to the folder where the CSV file will be saved
# csv_folder = os.path.join('results', criteria_group, criteria, version, f"{json_subfolder}-{test_number}")
# csv_filename = f'Annotation_of_{criteria_group}_{criteria}_prompt_version_{version}_test_number_{test_number}_model_{json_subfolder}_7b.csv'
csv_folder = os.path.join('results', dimension, version, f"{json_subfolder}-{test_number}")
csv_filename = f'Annotation_of_{dimension}_prompt_version_{version}_test_number_{test_number}_model_{json_subfolder}_7b.csv'
answers_file_path = os.path.join(csv_folder, csv_filename)


# ids_file_path = "./dimension_evaluation/list_ID.csv"

# Read the CSV files
answers_df = pd.read_csv(answers_file_path, header=0)
# ids_df = pd.read_csv(ids_file_path, header=0, sep=",")

print(answers_df['response'].dtype)  # Check the data type of the column
print(answers_df['response'].isnull().sum())  # Check for null values
print(answers_df['response'].head())  # Preview the first few rows

# Apply the function to extract grade information and create the "score" column
# answers_df["score"] = answers_df["response"].apply(extract_grade)
answers_df["score"] = answers_df["response"].apply(extract_numerical_grade)

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
