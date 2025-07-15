import os
import pandas as pd
import numpy as np
import torch
from transformers import AutoTokenizer, AutoModel
from sklearn.metrics.pairwise import cosine_similarity
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
import torch
from langdetect import detect, DetectorFactory
DetectorFactory.seed = 0  # To ensure consistent results

# Function to check if a text is in French
def is_french(text):
    try:
        return detect(text) == 'fr'  # Detect language and check if it is French
    except:
        return False  # In case of detection failure (e.g., empty string or gibberish)



# Set the device for processing (GPU if available)
print(torch.cuda.is_available())
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load French BERT model (or CamemBERT for better results with French)
model_name = "camembert-base"  # You can choose another model if needed
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModel.from_pretrained(model_name).to(device)

# Function to get embeddings for a given text
def get_embeddings(text):
    inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True, max_length=512).to(device)
    with torch.no_grad():
        outputs = model(**inputs)
    embeddings = outputs.last_hidden_state.mean(dim=1)  # Averaging over all tokens
    return embeddings.cpu().numpy()

# Function to calculate pairwise similarity
def calculate_similarity(texts_1, texts_2):
    embeddings_1 = np.vstack([get_embeddings(text) for text in texts_1])
    embeddings_2 = np.vstack([get_embeddings(text) for text in texts_2])
    similarities = cosine_similarity(embeddings_1, embeddings_2)
    return similarities

# Load question_formulation.csv
question_formulation = pd.read_csv('./question_formulation.csv')

# Placeholder to store results
similarities_results = []
explanation_issues = {'empty': 0, 'english': 0}

# For task 1: Analyze similarity between LLM explanations
def analyze_explanations_similarity(llm_outout_folder, criteria_group, criteria, version, test_number, json_subfolder):
    # Load the LLM output CSV file
    llm_output_file = f"Annotation_of_{criteria_group}_{criteria}_prompt_version_{version}_test_number_{test_number}_model_{json_subfolder}_7b.csv"
    llm_output_file=os.path.join(llm_outout_folder, llm_output_file)
    llm_data = pd.read_csv(llm_output_file)

    # Remove rows where explanations are empty or in English
    explanations = llm_data['explanations'].apply(lambda x: str(x))
    explanations = explanations[explanations.apply(is_french)]  # Keep only French explanations
    explanation_issues['english'] += (explanations.apply(lambda x: detect(x) != 'fr')).sum()
    explanations = explanations[explanations.str.strip() != '']  # Remove empty explanations
    explanation_issues['empty'] += (explanations.str.strip() == '').sum()

    # Filter the transcript IDs to match the valid explanations
    valid_transcript_ids = llm_data.loc[explanations.index, 'transcript ID']

    # Calculate pairwise similarity between explanations
    similarity_matrix = calculate_similarity(explanations, explanations)
    
    # Create a DataFrame to store the similarity results, using valid transcript IDs as index and columns
    similarity_df = pd.DataFrame(similarity_matrix, columns=valid_transcript_ids, index=valid_transcript_ids)
    
    # Save the similarity results
    similarity_df.to_csv(f'./explanation_analysis/similarity analysis/{criteria}/{version}/explanation_similarity_{criteria_group}_{criteria}_{version}_test_{test_number}.csv')
    
    # Plot the distribution as a boxplot
    plt.figure(figsize=(8, 6))
    sns.boxplot(data=similarity_matrix.flatten())
    plt.title(f'Similarity Distribution for {criteria_group} - {criteria}')
    plt.ylabel('Similarity Score')
    plt.savefig(f'./explanation_analysis/similarity analysis/{criteria}/{version}/explanation_similarity_{criteria_group}_{criteria}_{version}_test_{test_number}.png')
    plt.close()

# For task 2: Analyze similarity between explanations and criteria formulations
def analyze_criteria_similarity(llm_outout_folder, criteria_group, criteria, version, test_number, json_subfolder):
    # Load the LLM output CSV file
    llm_output_file = f"Annotation_of_{criteria_group}_{criteria}_prompt_version_{version}_test_number_{test_number}_model_{json_subfolder}_7b.csv"
    llm_output_file=os.path.join(llm_outout_folder, llm_output_file)
    llm_data = pd.read_csv(llm_output_file)
    
    # Get the criteria row from question_formulation.csv
    criteria_row = question_formulation[question_formulation['key_word'] == criteria].iloc[0]
    
    # Extract the content to compare against
    criteria_content = {
        'criteria_name': criteria_row['french'],
        'question': criteria_row['question'],
        'A': criteria_row['A'],
        'B': criteria_row['B'],
        'C': criteria_row['C']
    }
    

    # Remove rows where explanations are empty or in English
    explanations = llm_data['explanations'].apply(lambda x: str(x))
    print(len(explanations))

    explanations = explanations[explanations.apply(is_french)]  # Keep only French explanations
    print(len(explanations))
    
    explanations = explanations[explanations.str.strip() != '']  # Remove empty explanations
    print(len(explanations))

    # print(len(llm_data[llm_data['explanations'] is np.NaN]))
    # Calculate similarity between explanations and criteria
    similarities = {}
    for key, value in criteria_content.items():
        print(value)
        similarity_values= calculate_similarity(explanations, [str(value)] * len(explanations))
        # Flatten the result to ensure the value is 1D
        similarities[key] = similarity_values.flatten()
    # Save the similarity results
    similarity_df = pd.DataFrame(similarities)
    similarity_df.to_csv(f'./explanation_analysis/similarity analysis/{criteria}/{version}/criteria_explanation_similarity_{criteria_group}_{criteria}_{version}_test_{test_number}.csv')
    
    # Plot the similarity distributions as boxplots
    plt.figure(figsize=(8, 6))
    sns.boxplot(data=list(similarities.values()))  # Convert dict_values to list
    plt.xticks(ticks=np.arange(len(similarities)), labels=list(similarities.keys()))
    plt.title(f'Explanation Similarity to Criteria for {criteria_group} - {criteria}')
    plt.ylabel('Similarity Score')
    plt.savefig(f'./explanation_analysis/similarity analysis/{criteria}/{version}/criteria_explanation_similarity_{criteria_group}_{criteria}_{version}_test_{test_number}.png')
    plt.close()


# For task 3: Analyze similarity between explanations and corresponding transcript
def analyze_transcript_similarity(llm_outout_folder, criteria_group, criteria, version, test_number, json_subfolder):
    # Load the LLM output CSV file
    llm_output_file = f"Annotation_of_{criteria_group}_{criteria}_prompt_version_{version}_test_number_{test_number}_model_{json_subfolder}_7b.csv"
    llm_output_file=os.path.join(llm_outout_folder, llm_output_file)
    llm_data = pd.read_csv(llm_output_file)

    # Load the transcripts corresponding to each sample ID
    transcript_dir = './transcripts'
    transcript_similarity = []
    transcript_ids = []
    
    for _, row in llm_data.iterrows():
        transcript_id = row['transcript ID']
        explanation = row['explanations']
        
        # Skip empty or English explanations
        if not explanation or not is_french(explanation):
            continue
        
        transcript_file = os.path.join(transcript_dir, f'{transcript_id}.txt')
        if os.path.exists(transcript_file):
            with open(transcript_file, 'r', encoding='utf-8') as f:
                transcript_text = f.read()
                similarity = calculate_similarity([explanation], [transcript_text])[0][0]
                transcript_similarity.append(similarity)
                transcript_ids.append(transcript_id)
    
    # Save the similarity results
    similarity_df = pd.DataFrame({
        'transcript_id': transcript_ids,
        'similarity': transcript_similarity
    })
    similarity_df.to_csv(f'./explanation_analysis/similarity analysis/{criteria}/{version}/transcript_explanation_similarity_{criteria_group}_{criteria}_{version}_test_{test_number}.csv')
    
    # Plot the similarity distribution as a boxplot and a distribution plot
    plt.figure(figsize=(8, 6))
    sns.boxplot(x='transcript_id', y='similarity', data=similarity_df)
    plt.title(f'Transcript vs Explanation Similarity for {criteria_group} - {criteria}')
    plt.ylabel('Similarity Score')
    plt.savefig(f'./explanation_analysis/similarity analysis/{criteria}/{version}/transcript_explanation_similarity_box_{criteria_group}_{criteria}_{version}_test_{test_number}.png')
    plt.close()
    
    plt.figure(figsize=(8, 6))
    sns.histplot(similarity_df['similarity'], kde=True)
    plt.title(f'Transcript vs Explanation Similarity Distribution for {criteria_group} - {criteria}')
    plt.xlabel('Similarity Score')
    plt.ylabel('Frequency')
    plt.savefig(f'./explanation_analysis/similarity analysis/{criteria}/{version}/transcript_explanation_similarity_dist_{criteria_group}_{criteria}_{version}_test_{test_number}.png')
    plt.close()

# Run the analysis for each criteria group and criteria
criteria_group = 'storytelling'
criteria ='metaphor' # Add all other criteria here
version = '5'
test_number = '1'
json_subfolder = 'llama3'

llm_outout_folder =os.path.join("results", criteria_group, criteria, version, f"{json_subfolder}-{test_number}")

os.makedirs(f"./explanation_analysis/similarity analysis/{criteria}/{version}/", exist_ok=True)


analyze_explanations_similarity(llm_outout_folder, criteria_group, criteria, version, test_number, json_subfolder)
analyze_criteria_similarity(llm_outout_folder, criteria_group, criteria, version, test_number, json_subfolder)
analyze_transcript_similarity(llm_outout_folder, criteria_group, criteria, version, test_number, json_subfolder)

# Output the explanation issues to a text file
with open(f'./explanation_analysis/similarity analysis/{criteria}/{version}/explanation_issues.txt', 'w') as f:
    f.write(f"Empty explanations: {explanation_issues['empty']}\n")
    f.write(f"English explanations: {explanation_issues['english']}\n")

print("Analysis complete!")
