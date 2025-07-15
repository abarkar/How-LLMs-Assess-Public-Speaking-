import pandas as pd
from wordcloud import WordCloud
import matplotlib.pyplot as plt
import spacy
import os
from collections import defaultdict
import numpy as np

# Load French language model for NLP
nlp = spacy.load("fr_core_news_sm")

# Path to the folder containing JSON files
json_subfolder = 'llama3'
dimension = 'persuasiveness'
criteria_group = "storytelling"
criteria = "discours"
version = '5'
test_number = '1'
model_name_with_test_number = f"{json_subfolder}-{test_number}"
# Output folder
output_path = 'explanation_analysis'

# Path to the folder containing JSON files
json_parent_folder = os.path.join('terminaloutput', criteria_group, criteria, version, model_name_with_test_number)

# Path to the folder where the CSV file will be saved
csv_folder = os.path.join('results', criteria_group, criteria, version, f"{json_subfolder}-{test_number}")
csv_filename = f'Annotation_of_{criteria_group}_{criteria}_prompt_version_{version}_test_number_{test_number}_model_{json_subfolder}_7b.csv'

# Load the data
file_path = os.path.join(csv_folder, csv_filename)
data = pd.read_csv(file_path)

# Remove rows with NaN in the 'explanations' column
data = data.dropna(subset=["explanations"])

# Preprocessing function for French text
def preprocess_text(text):
    """
    Preprocess the text for word cloud generation.
    This includes tokenization, lemmatization, and removal of stopwords, punctuation, and non-alphanumeric tokens.
    """
    doc = nlp(text.lower())  # Convert to lowercase and tokenize
    tokens = [
        token.lemma_ for token in doc
        if not token.is_stop and token.is_alpha  # Remove stopwords and non-alphabetic tokens
    ]
    return " ".join(tokens)

# Apply preprocessing to the 'explanation' column
data["cleaned_explanation"] = data["explanations"].apply(preprocess_text)

# Group by 'score' and concatenate explanations for each category
grouped_explanations = defaultdict(str)
for _, row in data.iterrows():
    grouped_explanations[row["score"]] += row["cleaned_explanation"] + " "

# Function to generate a word cloud
def generate_word_cloud(text, title, output_file=None):
    """
    Generate and display a word cloud from the given text.
    """
    wordcloud = WordCloud(width=800, height=400, background_color="white").generate(text)
    plt.figure(figsize=(10, 5))
    plt.imshow(wordcloud, interpolation="bilinear")
    plt.axis("off")
    plt.title(title, fontsize=16)
    if output_file:
        plt.savefig(output_file, format="png")
    # plt.show()

# Generate word clouds for each score category
output_folder = os.path.join(output_path, criteria_group, criteria, version, model_name_with_test_number)
os.makedirs(output_folder, exist_ok=True)

for score, text in grouped_explanations.items():
    title = f"Word Cloud for Score {score}"
    output_file = os.path.join(output_folder, f"wordcloud_score_{score}.png")
    generate_word_cloud(text, title, output_file=output_file)

print("Word clouds generated and saved.")
