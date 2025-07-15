import os
import csv
import stanza
from collections import Counter

# Initialize Stanza pipeline for French
nlp = stanza.Pipeline(lang='fr', processors='tokenize,mwt,pos,lemma,depparse')

def is_declarative(sentence):
    return not (
        is_interrogative(sentence)
        or is_exclamatory(sentence)
        or is_imperative(sentence)
        or is_passive(sentence)
        or is_cleft(sentence)
        or is_conditional(sentence)
        or is_relative_clause(sentence)
        or is_nominal(sentence)
    )

def is_interrogative(sentence):
    return any(word.text == "?" or word.upos == "INT" for word in sentence.words)

def is_imperative(sentence):
    return any(
        word.deprel == "root"
        and word.upos == "VERB"
        and ("Mood=Imp" in word.feats if word.feats else False)
        for word in sentence.words
    )

def is_exclamatory(sentence):
    return any(word.text.endswith("!") for word in sentence.words)

def is_passive(sentence):
    return any("Voice=Pass" in word.feats if word.feats else False for word in sentence.words)

def is_cleft(sentence):
    tokens = [word.text.lower() for word in sentence.words]
    for i in range(len(tokens) - 1):
        if (tokens[i] == "c'" and tokens[i + 1] == "est") or (tokens[i] == "ce" and tokens[i + 1] == "qui"):
            return True
    return False

def is_conditional(sentence):
    return any(
        word.text.lower() == "si" or ("Mood=Cnd" in word.feats if word.feats else False)
        for word in sentence.words
    )

def is_relative_clause(sentence):
    return any(
        word.deprel == "acl:relcl" or word.text.lower() in {"qui", "que", "dont", "où"}
        for word in sentence.words
    )

def is_nominal(sentence):
    return not any(word.upos == "VERB" for word in sentence.words)

def classify_sentence(sentence):
    labels = []
    if is_interrogative(sentence):
        labels.append("Interrogative")
    if is_imperative(sentence):
        labels.append("Imperative")
    if is_exclamatory(sentence):
        labels.append("Exclamatory")
    if is_passive(sentence):
        labels.append("Passive")
    if is_cleft(sentence):
        labels.append("Cleft")
    if is_conditional(sentence):
        labels.append("Conditional")
    if is_nominal(sentence):
        labels.append("Nominal")
    if is_relative_clause(sentence):
        labels.append("Relative")
    if is_declarative(sentence):
        labels.append("Declarative")
    return labels

def analyze_text(text):
    doc = nlp(text)
    classifications = []

    for sentence in doc.sentences:
        classifications.extend(classify_sentence(sentence))

    total_sentences = len(doc.sentences)
    counts = Counter(classifications)
    percentages = {structure: (counts[structure] / total_sentences) * 100 for structure in counts}
    return percentages

# Data and feature locations
data_location = '/home/alisa/Documents/GitHubProjects/WillLLMsReplaceUs/data/transcripts'
feature_location = '/home/alisa/Documents/GitHubProjects/WillLLMsReplaceUs/features'
data_types = ['full']
directories = [os.path.join(data_location, f'{dtype}') for dtype in data_types]

# Ensure output directories exist
for dtype in data_types:
    output_dir = os.path.join(feature_location, dtype)
    os.makedirs(output_dir, exist_ok=True)

# Process each directory
for dir, dtype in zip(directories, data_types):
    output_csv = os.path.join(feature_location, dtype, 'syntactic_structures_count.csv')

    # Open the CSV file for writing
    with open(output_csv, 'w', newline='', encoding='utf-8') as csvfile:
        csvwriter = csv.writer(csvfile)
        # Write the header row
        header = [
            'category', 'Declarative', 'Relative', 'Interrogative', 'Exclamatory',
            'Imperative', 'Passive', 'Cleft', 'Conditional', 'Nominal'
        ]
        csvwriter.writerow(header)

        # Iterate over all text files in the directory
        for fname in sorted(os.listdir(dir)):
            if fname.endswith('.txt'):
                file_path = os.path.join(dir, fname)

                # Read the text content
                with open(file_path, 'r', encoding='utf-8') as file:
                    text = file.read()

                # Analyze text
                percentages = analyze_text(text)

                # Format category name: Remove directory prefix and ".txt" suffix
                category = os.path.basename(fname).replace('.txt', '')

                # Create a row for this file
                row = [category] + [percentages.get(key, 0.0) for key in header[1:]]
                csvwriter.writerow(row)
