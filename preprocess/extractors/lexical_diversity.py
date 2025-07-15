import math
from collections import Counter

def measure_of_textual_lexical_diversity(doc):
    lemmas = [token.lemma_ for token in doc.docFull]

    def calculate_mtld(lemmas, threshold=0.72):
        factors, start_index, types = 0, 0, Counter()
        for i, lemma in enumerate(lemmas):
            types[lemma] += 1
            ttr = len(types) / (i + 1 - start_index)
            if ttr < threshold:
                start_index = i + 1
                types.clear()
                factors += 1
        # Handle the last segment
        if types:
            ttr = len(types) / (len(lemmas) - start_index)
            factors += (1 - ttr) / (1 - threshold)
        return len(lemmas) / factors if factors else 0

    mtld_forward = calculate_mtld(lemmas)
    mtld_backward = calculate_mtld(list(reversed(lemmas)))

    mtld = (mtld_forward + mtld_backward) / 2
    return mtld if math.isfinite(mtld) else 0.0

def extract_features(doc):
    num_tokens = len(doc.docFull)
    num_types = len({token.pos_ for token in doc.docFull})
    lexical_diversity_features = dict()

    lexical_diversity_features["ttr"] = num_types / num_tokens if num_tokens else 0
    # lexical_diversity_features["corrected_ttr"] = num_types / math.sqrt(2 * num_tokens) if num_tokens else 0
    # lexical_diversity_features["root_ttr"] = num_types / math.sqrt(num_tokens) if num_tokens else 0
    # lexical_diversity_features["bilog_ttr"] = math.log(num_types) / math.log(num_tokens) if num_tokens > 1 else 0
    lexical_diversity_features["mtld"] = measure_of_textual_lexical_diversity(doc)

    return lexical_diversity_features
