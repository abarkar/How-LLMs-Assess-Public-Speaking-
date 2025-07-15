# def extract_features(doc):
#     # Initialize features
#     listOfFeatures = ['transitions_per_sentence', 'transitions_number', 'transitions_similarity_for_neighbour_sentenses', 'transition_types_count']
#     conjunctFeatures = dict.fromkeys(listOfFeatures, 0)

#     nbSent = len(doc.docPerSent)  # Number of sentences
#     docLen = len(doc.docFull)     # Length of the full document

#     conjunction = dict()  # To store conjunctions and their frequencies

#     # Iterate over each token in the full document
#     for token in doc.docFull:
#         # Count conjunctions
#         if token.pos_ in ["SCONJ", "CCONJ"]:
#             conjunctFeatures['transitions_number'] += 1
#             conjunction[token.lemma_] = conjunction.get(token.lemma_, 0) + 1
    
#     # Count sentence-level conjunction features
#     for curSent, sent in enumerate(doc.docPerSent):
#         if curSent < nbSent - 1:
#             if sent[0].text.lower() == doc.docPerSent[curSent + 1][0].text.lower():
#                 conjunctFeatures['transitions_similarity_for_neighbour_sentenses'] += 1

#     # Set unique transition types count
#     conjunctFeatures['transition_types_count'] = len(conjunction)

#     # Calculate the ratios
#     conjunctFeatures['transitions_per_sentence'] = conjunctFeatures['transitions_number'] / nbSent
#     conjunctFeatures['transitions_similarity_for_neighbour_sentenses'] = nbSent

#     return conjunctFeatures

import nltk
from nltk.corpus import stopwords

# Download the NLTK French stopwords
nltk.download('stopwords')
STOP_WORDS = set(stopwords.words("french"))

# Expletives list (French equivalents of emphasis words/phrases)
EXPLETIVES = {
    "en fait", "bien sûr", "évidemment", "je suppose", "je pense", "tu sais", 
    "vous voyez", "clairement", "certes", "remarquablement", "effectivement", 
    "en effet", "sans aucun doute"
}

# Coordinating conjunctions (for polysyndeton/asyndeton)
COORD_CONJUNCTIONS = {"et", "ou", "mais", "donc", "car", "ni"}
 
    
def extract_features(doc):
    # Initialize features
    listOfFeatures = [
        'transitions_per_sentence', 'transitions_number',
        'transitions_similarity_for_neighbour_sentenses', 'transition_types_count',
        'expletives_count', 'polysyndeton_count', 'asyndeton_count'
    ]
    conjunctFeatures = dict.fromkeys(listOfFeatures, 0)

    nbSent = len(doc.docPerSent)  # Number of sentences
    docLen = len(doc.docFull)     # Length of the full document

    conjunction = dict()  # To store conjunctions and their frequencies

    # Iterate over each token in the full document
    for token in doc.docFull:
        # Count conjunctions
        if token.pos_ in ["SCONJ", "CCONJ"]:
            conjunctFeatures['transitions_number'] += 1
            conjunction[token.lemma_] = conjunction.get(token.lemma_, 0) + 1

    # Count sentence-level features
    for curSent, sent in enumerate(doc.docPerSent):
        # Convert the sentence into a single string for phrase matching
        sentence_text = " ".join([token.text.lower() for token in sent])

        # Expletives: Check for occurrences of expletive phrases
        for phrase in EXPLETIVES:
            if phrase in sentence_text and not sentence_text.startswith(phrase) and not sentence_text.endswith(phrase):
                conjunctFeatures['expletives_count'] += 1

        if curSent < nbSent - 1:
            # Check transitions similarity between sentences
            if sent[0].text.lower() == doc.docPerSent[curSent + 1][0].text.lower():
                conjunctFeatures['transitions_similarity_for_neighbour_sentenses'] += 1

        # Detect Polysyndeton and Asyndeton within the sentence
        sentence_conjunctions = [token.text.lower() for token in sent if token.text.lower() in COORD_CONJUNCTIONS]
        if len(sentence_conjunctions) > 1:
            # Polysyndeton: Multiple coordinating conjunctions in succession
            conjunctFeatures['polysyndeton_count'] += sum(
                1 for i in range(len(sentence_conjunctions) - 1)
                if sentence_conjunctions[i] == sentence_conjunctions[i + 1]
            )
        # Asyndeton: Omitting conjunctions where they are expected
        elif len(sentence_conjunctions) == 0 and len(sent) > 1:
            # Check if a sentence has a list-like structure without conjunctions
            punctuations = {",", ";", ":"}
            conjunctFeatures['asyndeton_count'] += sum(
                1 for token in sent if token.text in punctuations
            )

    # Set unique transition types count
    conjunctFeatures['transition_types_count'] = len(conjunction)

    # Calculate the ratios
    conjunctFeatures['transitions_per_sentence'] = conjunctFeatures['transitions_number'] / nbSent
    conjunctFeatures['transitions_similarity_for_neighbour_sentenses'] /= nbSent

    return conjunctFeatures
