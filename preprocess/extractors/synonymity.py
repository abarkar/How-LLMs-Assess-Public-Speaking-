# # Alisa (16:26 16.01.2025): This function works perfectly
# from preprocess.disambiguation_module import disambiguate_synonyms
# from tqdm import tqdm
# import nltk
# from nltk.corpus import stopwords

# global similarity_threshold
# similarity_threshold=0.4

# nltk.download('stopwords')
# STOP_WORDS = set(stopwords.words("french"))


# def extract_features(doc):
#     """
#     Extract synonym usage with enhanced features for meaningful NOUNS, VERBS, and ADJECTIVES.

#     Features include:
#     - Noun_syn_diversification_potential (Count of synonyms for nouns above similarity threshold)
#     - Verb_syn_diversification_potential (Count of synonyms for verbs above similarity threshold)
#     - Adj_syn_diversification_potential (Count of synonyms for adjectives above similarity threshold)
#     - Noun_syn_diversification_relevance (Average similarity of synonyms for nouns)
#     - Verb_syn_diversification_relevance (Average similarity of synonyms for verbs)
#     - Adj_syn_diversification_relevance (Average similarity of synonyms for adjectives)
#     - Noun_synonimity (Average size of synonym groups for nouns)
#     - Verb_synonimity (Average size of synonym groups for verbs)

#     Args:
#         doc: Document object containing sentences and tokens.
#         similarity_threshold: Minimum similarity score to consider a synonym.

#     Returns:
#         Dictionary with calculated synonym features.
#     """
#     features = {
#         'Noun_syn_diversification_potential': 0,
#         'Verb_syn_diversification_potential': 0,
#         'Adj_syn_diversification_potential': 0,
#         'Noun_syn_diversification_relevance': 0,
#         'Verb_syn_diversification_relevance': 0,
#         'Adj_syn_diversification_relevance': 0,
#         'Noun_synonimity': 0,
#         'Verb_synonimity': 0
#     }

#     # Data structures to calculate group-based features
#     noun_groups = {}
#     verb_groups = {}
#     noun_tokens = set()
#     verb_tokens = set()

#     noun_similarities = []
#     verb_similarities = []
#     adj_similarities = []

#     # print("----- Processing document -----")

#     # for sentence in doc.docPerSent:
#     for token in doc.docFull:
#         if token.pos_ in {"NOUN", "VERB", "ADJ"} and token.text.isalpha() and token.text.lower() not in STOP_WORDS:
#             try:
#                 # Use disambiguate_synonyms to fetch synonyms
#                 synonyms = disambiguate_synonyms(token.text.lower(), token.sent, language="french")

#                 # Filter synonyms based on similarity threshold
#                 valid_synonyms = [syn for syn in synonyms if syn['similarity'] > similarity_threshold]
#                 similarities = [syn['similarity'] for syn in valid_synonyms]

#                 # Update POS-specific synonym potentials and average similarities
#                 if token.pos_ == "NOUN":
#                     features['Noun_syn_diversification_potential'] += len(valid_synonyms)
#                     noun_similarities.extend(similarities)
#                     noun_tokens.add(token.text.lower())

#                     # Group-based processing for nouns
#                     for syn in valid_synonyms:
#                         group = noun_groups.setdefault(syn['synonym'], [])
#                         group.append(token.text.lower())

#                 elif token.pos_ == "VERB":
#                     features['Verb_syn_diversification_potential'] += len(valid_synonyms)
#                     verb_similarities.extend(similarities)
#                     verb_tokens.add(token.text.lower())

#                     # Group-based processing for verbs
#                     for syn in valid_synonyms:
#                         group = verb_groups.setdefault(syn['synonym'], [])
#                         group.append(token.text.lower())

#                 elif token.pos_ == "ADJ":
#                     features['Adj_syn_diversification_potential'] += len(valid_synonyms)
#                     adj_similarities.extend(similarities)

#             except Exception as e:
#                 print(f"Error processing token '{token.text}': {e}")

#     # Remove duplicates from noun_tokens and verb_tokens
#     noun_tokens = set(noun_tokens)
#     verb_tokens = set(verb_tokens)

#     # Calculate average similarities
#     features['Noun_syn_diversification_relevance'] = (
#         sum(noun_similarities) / len(noun_similarities) if noun_similarities else 0
#     )
#     features['Verb_syn_diversification_relevance'] = (
#         sum(verb_similarities) / len(verb_similarities) if verb_similarities else 0
#     )
#     features['Adj_syn_diversification_relevance'] = (
#         sum(adj_similarities) / len(adj_similarities) if adj_similarities else 0
#     )

#     # Calculate group-based features
#     features['Noun_synonimity'] = (
#         sum(len(group) for group in noun_groups.values()) / len(noun_groups) if noun_groups else 0
#     )
#     features['Verb_synonimity'] = (
#         sum(len(group) for group in verb_groups.values()) / len(verb_groups) if verb_groups else 0
#     )

#     # print("----- Finished processing document -----")
#     # print("Synonym features:", features)

#     return features

import random
from preprocess.disambiguation_module import disambiguate_synonyms
from tqdm import tqdm
import nltk
from nltk.corpus import stopwords

global similarity_threshold
similarity_threshold = 0.4

nltk.download('stopwords')
STOP_WORDS = set(stopwords.words("french"))

def extract_features(doc):
    """
    Extract synonym usage with enhanced features for meaningful NOUNS, VERBS, and ADJECTIVES.

    Features include:
    - Noun_syn_diversification_potential (Count of synonyms for nouns above similarity threshold)
    - Verb_syn_diversification_potential (Count of synonyms for verbs above similarity threshold)
    - Adj_syn_diversification_potential (Count of synonyms for adjectives above similarity threshold)
    - Noun_syn_diversification_relevance (Average similarity of synonyms for nouns)
    - Verb_syn_diversification_relevance (Average similarity of synonyms for verbs)
    - Adj_syn_diversification_relevance (Average similarity of synonyms for adjectives)
    - Noun_synonimity (Average size of synonym groups for nouns)
    - Verb_synonimity (Average size of synonym groups for verbs)

    Args:
        doc: Document object containing sentences and tokens.
        similarity_threshold: Minimum similarity score to consider a synonym.

    Returns:
        Dictionary with calculated synonym features.
    """
    # Option to fix the number of random tokens to process
    num_random_tokens = 5  # Set to None to process all tokens, or specify an integer for random selection

    features = {
        'Noun_syn_diversification_potential': 0,
        'Verb_syn_diversification_potential': 0,
        'Adj_syn_diversification_potential': 0,
        'Noun_syn_diversification_relevance': 0,
        'Verb_syn_diversification_relevance': 0,
        'Adj_syn_diversification_relevance': 0,
        'Noun_synonimity': 0,
        'Verb_synonimity': 0
    }

    # Data structures to calculate group-based features
    noun_groups = {}
    verb_groups = {}
    noun_tokens = set()
    verb_tokens = set()

    noun_similarities = []
    verb_similarities = []
    adj_similarities = []

    # Select tokens to process
    all_nouns = [token for token in doc.docFull if token.pos_ == "NOUN"]
    all_verbs = [token for token in doc.docFull if token.pos_ == "VERB"]
    all_adjs = [token for token in doc.docFull if token.pos_ == "ADJ"]

    selected_nouns = random.sample(all_nouns, min(num_random_tokens, len(all_nouns))) if num_random_tokens else all_nouns
    selected_verbs = random.sample(all_verbs, min(num_random_tokens, len(all_verbs))) if num_random_tokens else all_verbs
    selected_adjs = random.sample(all_adjs, min(num_random_tokens, len(all_adjs))) if num_random_tokens else all_adjs

    tokens_to_process = selected_nouns + selected_verbs + selected_adjs

    for token in tokens_to_process:
        if token.pos_ in {"NOUN", "VERB", "ADJ"} and token.text.isalpha() and token.text.lower() not in STOP_WORDS:
            try:
                # Use disambiguate_synonyms to fetch synonyms
                synonyms = disambiguate_synonyms(token.text.lower(), token.sent, language="french")

                # Filter synonyms based on similarity threshold
                valid_synonyms = [syn for syn in synonyms if syn['similarity'] > similarity_threshold]
                similarities = [syn['similarity'] for syn in valid_synonyms]

                # Update POS-specific synonym potentials and average similarities
                if token.pos_ == "NOUN":
                    features['Noun_syn_diversification_potential'] += len(valid_synonyms)
                    noun_similarities.extend(similarities)
                    noun_tokens.add(token.text.lower())

                    # Group-based processing for nouns
                    for syn in valid_synonyms:
                        group = noun_groups.setdefault(syn['synonym'], [])
                        group.append(token.text.lower())

                elif token.pos_ == "VERB":
                    features['Verb_syn_diversification_potential'] += len(valid_synonyms)
                    verb_similarities.extend(similarities)
                    verb_tokens.add(token.text.lower())

                    # Group-based processing for verbs
                    for syn in valid_synonyms:
                        group = verb_groups.setdefault(syn['synonym'], [])
                        group.append(token.text.lower())

                elif token.pos_ == "ADJ":
                    features['Adj_syn_diversification_potential'] += len(valid_synonyms)
                    adj_similarities.extend(similarities)

            except Exception as e:
                print(f"Error processing token '{token.text}': {e}")

    # Remove duplicates from noun_tokens and verb_tokens
    noun_tokens = set(noun_tokens)
    verb_tokens = set(verb_tokens)

    # Calculate average similarities
    features['Noun_syn_diversification_relevance'] = (
        sum(noun_similarities) / len(noun_similarities) if noun_similarities else 0
    )
    features['Verb_syn_diversification_relevance'] = (
        sum(verb_similarities) / len(verb_similarities) if verb_similarities else 0
    )
    features['Adj_syn_diversification_relevance'] = (
        sum(adj_similarities) / len(adj_similarities) if adj_similarities else 0
    )

    # Calculate group-based features
    features['Noun_synonimity'] = (
        sum(len(group) for group in noun_groups.values()) / len(noun_groups) if noun_groups else 0
    )
    features['Verb_synonimity'] = (
        sum(len(group) for group in verb_groups.values()) / len(verb_groups) if verb_groups else 0
    )

    return features
