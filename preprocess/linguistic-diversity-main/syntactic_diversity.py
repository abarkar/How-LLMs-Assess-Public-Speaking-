import os
# Set CUDA device
os.environ["CUDA_VISIBLE_DEVICES"] = "1"
import torch
import random
import numpy as np
from tqdm import tqdm
import networkx as nx
from grakel.utils import graph_from_networkx
from grakel.kernels import WeisfeilerLehman, VertexHistogram
import nltk
from nltk.tokenize import sent_tokenize
import stanza
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
from scipy.stats import zscore
import scipy.stats as stats
import csv
print(torch.cuda.is_available())  # Should return True if CUDA is available

# # Download and initialize Stanza pipeline for NLP processing
# stanza.download('en')
# nlp = stanza.Pipeline('en', processors='tokenize,pos,mwt,lemma,depparse')


# Alisa: Changed to french parser
# Download and initialize Stanza pipeline for French NLP processing
stanza.download('fr')
nlp = stanza.Pipeline('fr', processors='tokenize,pos,mwt,lemma,depparse')

# Alisa: nltk.sent_tokenize uses a language-specific tokenizer.
#           By default, it uses English rules, so you need to 
#           specify the French tokenizer
# Download the French sentence tokenizer
nltk.download('punkt')
nltk.download('punkt_tab')

# Function to calculate the mean confidence interval
def mean_confidence_interval(data, confidence=0.95):
    data = np.array(data)
    mean = np.mean(data)
    sem = stats.sem(data)  # Standard error of the mean
    margin_of_error = sem * stats.norm.ppf((1 + confidence) / 2.)
    return mean, margin_of_error

# Function to create a dependency graph from a Stanza-processed sentence
def create_graph(doc):
    sent = doc.sentences[0]
    G = nx.Graph()

    for word in sent.to_dict():
        if isinstance(word['id'], tuple):
            continue

        # Add the current word to the graph
        if word['id'] not in G.nodes():
            G.add_node(word['id'])
            
        G.nodes[word['id']]['label'] = word['upos']

        # Add the head of the word (the word it depends on)
        if word['head'] not in G.nodes():
            G.add_node(word['head'])

        G.add_edge(word['id'], word['head'])

    # Set the root node label
    G.nodes[0]['label'] = 'none'
    
    return G

# Parameters
n = 20  # Number of sentences to samples
# output_file = 'syn.txt'

feature_location='/home/alisa/Documents/GitHubProjects/WillLLMsReplaceUs/features'
data_location='/home/alisa/Documents/GitHubProjects/WillLLMsReplaceUs/data/transcripts/'

# feature_location='/home/alisa/Documents/GitHubProjects/WhatIsPErsuasiveTextFromLLMsPointOfView/preprocess/linguistic-diversity-main/features'
# data_location='/home/alisa/Documents/GitHubProjects/WhatIsPErsuasiveTextFromLLMsPointOfView/preprocess/linguistic-diversity-main'


# List of directories to process
data_types = ['end', 'middle', 'beginning']
directories=[os.path.join(data_location, f'{dtype}') for dtype in data_types]

print("directories", directories)

# Process each directory
for dir, dtype in zip(directories, data_types):
    print(dir, dtype)

    output_csv = os.path.join(feature_location, dtype, 'syntactic_diversity.csv')

    # Open the CSV file for writing
    with open(output_csv, 'w', newline='', encoding='utf-8') as csvfile:
        csvwriter = csv.writer(csvfile)
        # Write the header row
        csvwriter.writerow(['category', 'syntactic_diversity_mean', 'syntactic_diversity_error'])

        # Get list of files in the directory
        files_name = [fname for fname in os.listdir(dir) if os.path.isfile(os.path.join(dir, fname))]
        files = [os.path.join(dir, fname) for fname in files_name]

        # Process each file
        for fname in files:
            print(fname)
            graphs = []

            with open(fname) as f:
                lines = f.readlines()
                lines = [l.replace('<newline>', '\n') for l in lines if l.strip() != ""]

                # Tokenize sentences
                sentences = []
                for t in lines:
                    # Alisa: added french tokenizer which is not default
                    sents = sent_tokenize(t, language='french')
                    # print("---------------------------------")
                    # print("line", t, "\n")
                    # print("\n *** \n")
                    # print("sents", sents, "\n")
                    # print("---------------------------------")
                    
                    # Alisa: In our case all the sentences should be complete, therefore, no need to cut.
                    sentences += sents

                    # #ideally discard the first and last sentences because they might not be complete
                    # if len(sents) > 2:
                    #     sentences += sents[1:-1]
                    # elif len(sents) > 1:
                    #     sentences += [sents[0]]
                    # else:
                    #     sentences += sents
                # Randomly sample sentences
                print(len(sentences))
                random.seed(42)
                sentences = random.sample(sentences, min(len(sentences), n))

                # Process each sentence and create graph
                for s in tqdm(sentences):
                    doc = nlp(s)
                    graphs.append(create_graph(doc))

            # Convert NetworkX graphs to Grakel format
            G = list(graph_from_networkx(graphs, node_labels_tag='label'))

            # Initialize Weisfeiler-Lehman kernel
            gk = WeisfeilerLehman(n_iter=2, normalize=True, base_graph_kernel=VertexHistogram)

            # Compute kernel matrix
            K = gk.fit_transform(G)
            # K = torch.tensor(K).to(torch.float).to("cuda:0")
            K = torch.tensor(K).to(torch.float)  # This will default to using the CPU


            # Extract non-diagonal elements from kernel matrix
            mask = ~torch.eye(K.size(0), dtype=bool)
            non_diag_elements = K[mask]
            non_diag_array = non_diag_elements.cpu().numpy()

            # Rescale values and compute mean and confidence interval
            res = 1 - non_diag_array
            mean, error = mean_confidence_interval(res)


            # Format category name: Remove directory prefix and ".txt" suffix
            category = os.path.basename(fname).replace('.txt', '')

            # Write results to the CSV file
            csvwriter.writerow([category, mean, error])

            # Clear GPU memory
            torch.cuda.empty_cache()

# Done processing all files
print(f'Results written to {output_csv}')
