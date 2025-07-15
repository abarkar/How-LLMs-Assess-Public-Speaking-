def extract_features(doc):
    """Calculate frequency of passive constructions."""
    passive_voice_features = {'passive_voice_count': 0}
    for token in doc.docFull:
            # print(f"Token: {token.text}, POS: {token.pos_}, Dep: {token.dep_}")
            if token.dep_ == "aux:pass":
                passive_voice_features['passive_voice_count'] += 1
    return passive_voice_features
