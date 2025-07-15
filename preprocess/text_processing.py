import os
import re
import pandas as pd
import codecs
import stanza


# Ensure Stanza is set up for French

stanza.download("fr")
nlp = stanza.Pipeline(lang="fr", processors="tokenize,pos,lemma,depparse")

# Define token class for French
class frencToken:
    def __init__(self, sent=None, text=None, lemma=None, pos=None, dep=None):
        self.sent = sent
        self.text = text
        self.lemma_ = lemma
        self.pos_ = pos
        self.dep_ = dep # Alisa: what is this for?
        # self.tag_ = tag

# Define document class for French
class frenchDoc:
    def __init__(self, transcript, listOfFrenchSentences):
        self.docFull = []
        self.text = transcript
        for sent in listOfFrenchSentences:
            self.docFull.extend(sent)
        self.docPerSent = listOfFrenchSentences

# Function to perform full tagging for French
def fullTaggingForFrench(transcript, dataset, root_path):

    # Use regex to split by sentence-ending punctuation
    sentences = [s.strip() for s in re.split(r"(?<=[.!?])\s+", transcript) if s.strip()]

    listOfSentences = []
    for sentence in sentences:
        stanza_doc = nlp(sentence)
        stanza_tokens = []
        for stanza_sentence in stanza_doc.sentences:
            for word in stanza_sentence.words:
                stanza_tokens.append(frencToken(
                    sent=stanza_sentence.text,
                    text=word.text,
                    lemma=word.lemma,
                    pos=word.upos,
                    dep=word.deprel #,
                    # tag=None  # Stanza does not provide a direct "tag" equivalent
                ))

        listOfSentences.append(stanza_tokens)

    doc = frenchDoc(transcript, listOfSentences)

    # Save the results to a CSV file
    output_dir = os.path.join(root_path, f"features/{dataset}/")
    os.makedirs(output_dir, exist_ok=True)
    result_file = os.path.join(output_dir, "full_tagging_results.csv")
    data = [
        [token.text, token.lemma_, token.pos_, token.dep_]
        for sentence in listOfSentences
        for token in sentence
    ]
    df = pd.DataFrame(data, columns=["Word", "Lemma", "POS", "Dependency"])
    df.to_csv(result_file, index=False)

    return doc

# Preprocess raw transcript and split it into sentences
def text2sentence(key, dataset, root_path):
    dir_path = os.path.join(root_path, "data", "transcripts", f"{dataset}")
    file_path = os.path.join(dir_path, key)

    with codecs.open(file_path, "r", "utf-8") as f:
        lines = f.readlines()
        text = ""
        for line in lines:
            tmp = line.strip()
            if len(tmp) != 0:
                text += tmp + " "

    doc = fullTaggingForFrench(text, dataset, root_path)
    return doc