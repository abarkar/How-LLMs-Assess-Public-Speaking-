# def extract_features(doc):
#     # Initialize the overlap features with updated names and removed features
#     features = ['local_NOUN_overlap', 'local_content_overlap', 
#                 'global_NOUN_overlap', 'global_content_overlap']
#     overlapFeatures = dict.fromkeys(features, 0)

#     totalSentences = len(doc.docPerSent)
#     # print("totalSentences", totalSentences)

#     for i in range(totalSentences):
#         for j in range(i + 1, totalSentences):
#             sentence1 = doc.docPerSent[i]
#             sentence2 = doc.docPerSent[j]
#             # Convert sentences to lemma sets
#             sentence1_lemmas = {word.lemma_ for word in sentence1 if word.pos_}
#             sentence2_lemmas = {word.lemma_ for word in sentence2 if word.pos_}

#             # Noun overlap
#             noun_overlap = {word.lemma_ for word in sentence1 if word.pos_ == "NOUN"} & \
#                            {word.lemma_ for word in sentence2 if word.pos_ == "NOUN"}

#             # Content word overlap
#             content_overlap = sentence1_lemmas & sentence2_lemmas - \
#                               {word.lemma_ for word in sentence1 if word.pos_ == "PRON"}

#             # Update features
#             if noun_overlap:
#                 overlapFeatures['global_NOUN_overlap'] += 1
#                 if j - i == 1:
#                     overlapFeatures['local_NOUN_overlap'] += 1

#             if content_overlap:
#                 overlapFeatures['global_content_overlap'] += len(content_overlap)
#                 if j - i == 1:
#                     overlapFeatures['local_content_overlap'] += len(content_overlap)

#     # Normalize features
#     overlapFeatures = {key: value / totalSentences for key, value in overlapFeatures.items()}
#     return overlapFeatures
    
import nltk
from nltk.corpus import stopwords

# Download the NLTK French stopwords
nltk.download('stopwords')
STOP_WORDS = set(stopwords.words("french"))

def extract_features(doc):
    # Initialize the overlap features
    features = [
        'local_NOUN_overlap', 'local_content_overlap',
        'global_NOUN_overlap', 'global_content_overlap',
        'alliteration', 'anaphora', 'antimetabole', 'epanalepsis'
    ]
    overlapFeatures = dict.fromkeys(features, 0)

    totalSentences = len(doc.docPerSent)

    def clean_words(sentence):
        """
        Filters out stopwords and words with unwanted POS tags (PRON, conjunctions).
        """
        return [
            token.text.lower() for token in sentence
            if token.text.lower() not in STOP_WORDS and token.pos_ not in {"PRON", "CCONJ", "SCONJ"}
        ]

    for i in range(totalSentences):
        sentence1 = doc.docPerSent[i]
        words1 = clean_words(sentence1)

        # Local and Global Overlap Features
        for j in range(i + 1, totalSentences):
            sentence2 = doc.docPerSent[j]
            words2 = clean_words(sentence2)

            # Convert sentences to lemma sets
            sentence1_lemmas = {token.lemma_ for token in sentence1 if token.pos_}
            sentence2_lemmas = {token.lemma_ for token in sentence2 if token.pos_}

            # Noun overlap
            noun_overlap = {token.lemma_ for token in sentence1 if token.pos_ == "NOUN"} & \
                           {token.lemma_ for token in sentence2 if token.pos_ == "NOUN"}

            # Content word overlap
            content_overlap = sentence1_lemmas & sentence2_lemmas - \
                              {token.lemma_ for token in sentence1 if token.pos_ in {"PRON", "CCONJ", "SCONJ"}}

            # Update overlap features
            if noun_overlap:
                overlapFeatures['global_NOUN_overlap'] += len(noun_overlap)
                if j - i == 1:
                    overlapFeatures['local_NOUN_overlap'] += len(noun_overlap)

            if content_overlap:
                overlapFeatures['global_content_overlap'] += len(content_overlap)
                if j - i == 1:
                    overlapFeatures['local_content_overlap'] += len(content_overlap)

        # Alliteration within 0-2 word distances
        for k in range(len(words1) - 1):
            for d in range(1, 3):  # Check distances of 1 and 2
                if k + d < len(words1) and words1[k][0] == words1[k + d][0]:
                    overlapFeatures['alliteration'] += 1

        # Anaphora: Phrase repetition across clauses and sentences
        anaphora_lengths = []
        for start_idx in range(len(words1)):
            for end_idx in range(start_idx + 1, len(words1) + 1):
                phrase = words1[start_idx:end_idx]
                phrase_len = len(phrase)
                if phrase_len < 1:
                    continue
                # Check for repetition in the same sentence
                for check_start in range(end_idx, len(words1) - phrase_len + 1):
                    if words1[check_start:check_start + phrase_len] == phrase:
                        anaphora_lengths.append(phrase_len)
                        break
                # Check for repetition in the next sentence
                if i + 1 < totalSentences:
                    words2 = clean_words(doc.docPerSent[i + 1])
                    for check_start in range(len(words2) - phrase_len + 1):
                        if words2[check_start:check_start + phrase_len] == phrase:
                            anaphora_lengths.append(phrase_len)
                            break
        if anaphora_lengths:
            overlapFeatures['anaphora'] = sum(anaphora_lengths) / len(anaphora_lengths)

        # Antimetabole: Reversed phrase repetition across clauses/sentences
        antimetabole_lengths = []
        for start_idx in range(len(words1)):
            for end_idx in range(start_idx + 1, len(words1) + 1):
                phrase = words1[start_idx:end_idx]
                phrase_len = len(phrase)
                if phrase_len < 1:
                    continue
                # Check for reversed repetition in the current sentence
                reversed_words1 = words1[::-1]
                for check_start in range(len(reversed_words1) - phrase_len + 1):
                    if reversed_words1[check_start:check_start + phrase_len] == phrase:
                        antimetabole_lengths.append(phrase_len)
                        break
                # Check for reversed repetition in the next sentence
                if i + 1 < totalSentences:
                    words2 = clean_words(doc.docPerSent[i + 1])
                    reversed_words2 = words2[::-1]
                    for check_start in range(len(reversed_words2) - phrase_len + 1):
                        if reversed_words2[check_start:check_start + phrase_len] == phrase:
                            antimetabole_lengths.append(phrase_len)
                            break
        if antimetabole_lengths:
            overlapFeatures['antimetabole'] = sum(antimetabole_lengths) / len(antimetabole_lengths)

        # Epanalepsis: First 3 words vs last 3 words
        first_three_words = [token.text.lower() for token in sentence1[:3]]
        last_three_words = [token.text.lower() for token in sentence1[-3:]]
        overlapFeatures['epanalepsis'] += any(
            word in last_three_words for word in first_three_words
        )

    # Normalize features
    for key in ['local_NOUN_overlap', 'local_content_overlap','global_NOUN_overlap', 'global_content_overlap', 'epanalepsis']:
        overlapFeatures[key] /= totalSentences
    overlapFeatures['alliteration'] /= len(doc.docFull)
    return overlapFeatures
