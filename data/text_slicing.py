# ./data/split_transcripts_by_thirds.py

import os
import spacy
from pathlib import Path

# Load French sentence segmenter
nlp = spacy.load("fr_core_news_sm")

# Paths
base_dir = Path(__file__).parent
input_dir = base_dir / "transcripts/full"
output_dirs = {
    "beginning": base_dir / "transcripts/beginning",
    "middle": base_dir / "transcripts/middle",
    "end": base_dir / "transcripts/end"
}

# Ensure output directories exist
for path in output_dirs.values():
    path.mkdir(parents=True, exist_ok=True)

def normalize_text(text):
    """
    Remove line breaks inside paragraphs, normalize spaces.
    """
    text = text.replace('\n', ' ')
    text = ' '.join(text.split())  # collapse multiple spaces
    return text

# Function to split a list into three nearly equal parts
def split_into_thirds(sentences):
    n = len(sentences)
    third = n // 3
    remainder = n % 3
    return (
        sentences[:third + (1 if remainder > 0 else 0)],
        sentences[third + (1 if remainder > 0 else 0): 2*third + (1 if remainder > 1 else 0)],
        sentences[2*third + (1 if remainder > 1 else 0):]
    )

# Process each transcript
for txt_file in input_dir.glob("*.txt"):
    with open(txt_file, "r", encoding="utf-8") as f:
        raw_text = f.read()

    clean_text = normalize_text(raw_text)
    doc = nlp(clean_text)
    sentences = [sent.text.strip() for sent in doc.sents if sent.text.strip()]

    # If there are fewer than 3 sentences, skip
    if len(sentences) < 3:
        print(f"Skipping {txt_file.name} (too few sentences)")
        continue

    # Split into thirds
    beg, mid, end = split_into_thirds(sentences)
    thirds = {"beginning": beg, "middle": mid, "end": end}

    for part, sents in thirds.items():
        out_path = output_dirs[part] / txt_file.name
        with open(out_path, "w", encoding="utf-8") as f:
            f.write("\n".join(sents))
