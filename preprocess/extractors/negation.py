def extract_features(doc):
    """Calculate the frequency of negation words, avoiding double-counting for paired negations."""
    negation_features = {'negation_count': 0}
    negation_starters = {"ne", "n'"}  # Starting part of negation
    negation_complements = {"pas", "jamais", "aucun", "ni", "rien"}  # Complementary part of negation

    for sent in doc.docPerSent:
        in_negation = False  # Track if we're inside a negation structure
        for i, token in enumerate(sent):
            if token.text in negation_starters:
                in_negation = True  # Mark the start of a negation
            elif in_negation and token.text in negation_complements:
                # If a complement follows a starter, count as one negation
                negation_features['negation_count'] += 1
                in_negation = False  # Reset after counting the pair
            elif token.text in negation_complements and not in_negation:
                # Standalone complements (e.g., "jamais" without "ne")
                negation_features['negation_count'] += 1
                in_negation = False  # Ensure we don't incorrectly double-count

        # Reset for each sentence
        in_negation = False

    return negation_features
