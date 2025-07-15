def extractSyllableMetrics(doc):
    """
    Calculate syllable-based features for French text.
    """
    syllable_features = {'average_syllables': 0, 'polysyllabic_word_count': 0}
    total_syllables = 0
    polysyllabic_count = 0
    total_words = 0
    
    for token in doc.docFull:
        # Use the custom French syllable counter
        syllables = count_syllables_french(token.text)
        total_syllables += syllables
        total_words += 1
        if syllables > 2:
            polysyllabic_count += 1

    # Calculate average syllables per word
    syllable_features['average_syllables'] = total_syllables / total_words if total_words > 0 else 0
    syllable_features['polysyllabic_word_count'] = polysyllabic_count
    
    return syllable_features


def count_syllables_french(word):
    """
    Estimate the number of syllables in a French word.
    """
    vowels = "aeiouyâàäéèêëîïôöûüù"
    word = word.lower()
    syllable_count = 0
    previous_char_is_vowel = False

    for char in word:
        if char in vowels:
            # Avoid counting consecutive vowels as multiple syllables
            if not previous_char_is_vowel:
                syllable_count += 1
            previous_char_is_vowel = True
        else:
            previous_char_is_vowel = False

    # French-specific rules:
    if word.endswith(('e', 'es', 'ent')) and len(word) > 2:
        syllable_count -= 1

    return max(1, syllable_count)  # Ensure at least one syllable

def flesch_kincaid_score_spacy(doc):
    """
    Calculate Flesch-Kincaid Reading Ease Score for French text.
    """
    
    
    # Extract sentences and words
    words = [token.text for token in doc.docFull if token.text.isalpha()]
    
    # Count syllables
    syllables = sum(count_syllables_french(word) for word in words)
    
    # Calculate metrics
    asl = len(words) / len(doc.docPerSent)  # Average sentence length
    asw = syllables / len(words)  # Average syllables per word
    
    # Flesch Reading Ease formula for French
    flesch_score = 206.835 - (1.015 * asl) - (84.6 * asw)
    return flesch_score

def gunning_fog_index_spacy(doc):
    """
    Calculate Gunning Fog Index for French text.
    """
    # Process text using spaCy
    # doc = nlp(text)
    
    # Extract sentences and words
    # sentences = list(doc.sents)
    # print("nymber of sentences:", len(doc.docPerSent))
    alphabetic_words = [token for token in doc.docFull if token.text.isalpha()]
    
    # Count complex words (3 or more syllables) and not proper pronouns
    complex_words = [word.text for word in alphabetic_words if count_syllables_french(word.text) >= 3 and word.pos_ != "PROPN" ]
    
    # Calculate metrics
    asl = len(alphabetic_words) / len(doc.docPerSent)  # Average sentence length
    phw = (len(complex_words) / len(alphabetic_words)) * 100  # Percentage of hard words
    
    # Gunning Fog Index formula
    fog_index = 0.4 * (asl + phw)
    return fog_index

def extract_features(doc):
    """Calculate readability metrics."""
    # text = " ".join([token.text for token in doc.docFull])
    syllable_features=extractSyllableMetrics(doc)
    readability_features = {
        'flesch_kincaid_score': flesch_kincaid_score_spacy(doc),
        'gunning_fog_index': gunning_fog_index_spacy(doc),
        'average_syllables':syllable_features['average_syllables'],
        'polysyllabic_word_count':syllable_features['polysyllabic_word_count']
    }
    return readability_features