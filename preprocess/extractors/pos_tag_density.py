from collections import Counter

ALL_POS_TAGS = ['ADJ', 'ADV', 'INTJ', 'NOUN', 'PRON', 'PROPN', 'VERB', 'AUX',
    'DET', 'ADP', 'CCONJ', 'SCONJ', 'NUM', 'PART', 'SYM', 'X']

def extract_features(doc):
    """Calculate the proportions of word classes, including missing classes as 0."""
    # Count occurrences of each POS in the text
    word_classes = Counter(token.pos_ for token in doc.docFull)
    
    # Total number of tokens to calculate proportions
    total_tokens = sum(word_classes.values())
    
    # Initialize features with 0 for all POS tags
    word_class_features = {f"proportion_{pos}": 0 for pos in ALL_POS_TAGS}
    
    # Calculate the proportions for the present POS tags
    if total_tokens > 0:
        for pos, count in word_classes.items():
            if pos in ALL_POS_TAGS :
                word_class_features[f"proportion_{pos}"] = count / total_tokens
    
    return word_class_features
