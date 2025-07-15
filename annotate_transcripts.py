
import os
import openai
import pandas as pd
import time
from tqdm import tqdm
import csv


# ---------------------- CONFIGURATION ----------------------
MODEL_NAME = "gpt-4.1"  # Change here to switch models
PROMPTING = 'few-shot'
TRANSCRIPTS_DIR = "./data/transcripts/full"
EXPERT_CSV = "./data/expert_annotations.csv"
MC_FORMULATIONS = "./MC_formulations.csv"
LIKERT_FORMULATIONS = "./Likert_formulations.csv"
MODEL_OUTPUT_DIR = f"./model_output/{MODEL_NAME}/{PROMPTING}"
FINAL_ANNOTATION_OUTPUT = f"./data/{MODEL_NAME}_{PROMPTING}_annotations.csv"


# Ensure output folder exists
os.makedirs(MODEL_OUTPUT_DIR, exist_ok=True)

# ---------------------- LOAD DATA ----------------------

df_expert = pd.read_csv(EXPERT_CSV)
mc_formulations = pd.read_csv(MC_FORMULATIONS)
likert_formulations = pd.read_csv(LIKERT_FORMULATIONS)

# Use French column for criterion names
criteria_columns = df_expert.columns[1:-2]  # excluding ID, Comment, Annotation time
mc_criteria = set(mc_formulations['French'])
likert_criteria = set(likert_formulations['French'])

# ---------------------- FUNCTIONS ----------------------

def load_transcript(transcript_id):
    path = os.path.join(TRANSCRIPTS_DIR, f"{transcript_id}.txt")
    with open(path, "r", encoding="utf-8") as file:
        return file.read().replace('\n', ' ').strip()

def construct_prompt(transcript, question_data, is_mc=True, examples=None):
    if is_mc:
        if PROMPTING=='zero-shot':
            prompt = f"""TRANSCRIPT={transcript}
                TASK=[Based on the provided transcript of the public performance, choose one of the options that best answers the following question.]

                Question: {question_data['question']}
                Answer options:
                A: {question_data['A']}
                B: {question_data['B']}
                C: {question_data['C']}

                Respond using the format:

                Note: <MASK>

                Where <MASK> is A, B, or C - corresponding to the chosen option.
                Provide no other information beyond this answer."""
        elif PROMPTING=='few-shot':
            if not examples:
                print(f"Missing examples.")
                

            prompt = f"""TRANSCRIPT={transcript}
                TASK=[Based on the provided transcript of the public performance, choose one of the options that best answers the following question. Use examples of expert question answering to guide you.]
                EXAMPLE_1 = [Transcript: {examples['A']['transcript']} Expert evaluation of this transcript is: {examples['A']['evaluation']}]
                EXAMPLE_2 = [Transcript: {examples['B']['transcript']} Expert evaluation of this transcript is: {examples['B']['evaluation']}]
                EXAMPLE_3 = [Transcript: {examples['C']['transcript']} Expert evaluation of this transcript is: {examples['C']['evaluation']}]
                
                Now evaluate this transcript by answering to the provided question:
                TRANSCRIPT={transcript}
                Question: {question_data['question']}
                Answer options:
                A: {question_data['A']}
                B: {question_data['B']}
                C: {question_data['C']}

                Respond using the format:

                Note: <MASK>

                Where <MASK> is A, B, or C - corresponding to the chosen option.
                Provide no other information beyond this answer."""
    else:
        if PROMPTING=='zero-shot':
            prompt = f"""TRANSCRIPT={transcript}
                TASK=[Based on the transcript, respond to the question by assigning a score (natural number) according to the provided scale.]

                Question: {question_data['question']}
                Response scale: {question_data['scale']}

                Respond using the format:

                Note: <MASK>

                Where <MASK> is your score from the response scale.
                Provide no other information beyond this answer."""
        elif PROMPTING=='few-shot':
            if not examples:
                print(f"Missing examples.")
            
            prompt = f"""
                TASK=[Based on the transcript, respond to the question by assigning a score (natural number) according to the provided scale. Use examples of expert question answering to guide you.]
                EXAMPLE_1 = [Transcript: {examples['min']['transcript']} Expert evaluation of this transcript is: {examples['min']['evaluation']}]
                EXAMPLE_3 = [Transcript: {examples['max']['transcript']} Expert evaluation of this transcript is: {examples['max']['evaluation']}]
                
                Now evaluate this transcript by assigning the score to the provided question:
                TRANSCRIPT={transcript}
                Question: {question_data['question']}
                Response scale: {question_data['scale']}

                Respond using the format:

                Note: <MASK>

                Where <MASK> is your score from the response scale.
                Provide no other information beyond this answer."""

    return prompt

def call_model(prompt):
    client = openai.OpenAI(api_key="OPEN_API_KEY")
    start = time.time()

    completion = client.chat.completions.create(
        model=MODEL_NAME,
        messages=[{"role": "user", "content": prompt}]
    )

    duration = time.time() - start
    content = completion.choices[0].message.content.strip()
    tokens = completion.usage.total_tokens

    return content, duration, tokens, MODEL_NAME

def extract_masked_note(response):
    for line in response.splitlines():
        if line.strip().lower().startswith("note:"):
            return line.split(":")[-1].strip()
    return "N/A"


def form_example(criterion, is_mc, current_id):
    examples = {}

    if is_mc:
        mc_row = mc_formulations[mc_formulations['French'] == criterion].iloc[0].to_dict()
        for option in ['A', 'B', 'C']:
            matching = df_expert[(df_expert[criterion] == option) & (df_expert['ID'] != current_id)]
            if not matching.empty:
                row = matching.sample(1).iloc[0]
                transcript = load_transcript(row['ID'])
                evaluation = mc_row[option]
            else:
                transcript = "No transcript annotated as such by an expert."
                evaluation = mc_row[option]
            examples[option] = {
                'transcript': transcript,
                'evaluation': evaluation
            }

    else:
        likert_values = df_expert[criterion].dropna().unique()
        if len(likert_values) == 0:
            print(f"No valid values for criterion: {criterion}")
            return {}
        min_score = min(likert_values)
        max_score = max(likert_values)

        for label, score in [('min', min_score), ('max', max_score)]:
            matching = df_expert[(df_expert[criterion] == score) & (df_expert['ID'] != current_id)]
            if not matching.empty:
                row = matching.sample(1).iloc[0]
                transcript = load_transcript(row['ID'])
            else:
                transcript = "No transcript annotated as such by an expert."
            examples[label] = {
                'transcript': transcript,
                'evaluation': str(score)
            }

    return examples


# ---------------------- MAIN SCRIPT ----------------------

df_gpt = df_expert[['ID']].copy()

for criterion in tqdm(criteria_columns, desc="Processing criteria"):
    print(f"→ {criterion}")

    is_mc = criterion in mc_criteria

    if is_mc:
        form_row = mc_formulations[mc_formulations['French'] == criterion].iloc[0].to_dict()
    elif criterion in likert_criteria:
        form_row = likert_formulations[likert_formulations['French'] == criterion].iloc[0].to_dict()
    else:
        print(f"⚠️ Criterion not found in formulation files: {criterion}")
        continue

    gpt_responses = []
    full_model_log = []

    for _, row in tqdm(df_expert.iterrows(), total=len(df_expert), desc=f"→ {criterion}", leave=False):
        transcript_id = row['ID']
        try:
            transcript = load_transcript(transcript_id)
        except FileNotFoundError:
            print(f"Transcript not found: {transcript_id}. Skipping.")
            gpt_responses.append("MISSING")
            full_model_log.append({
                "ID": transcript_id,
                "Model Output": "FILE_NOT_FOUND",
                "Model": MODEL_NAME,
                "Time Spent (s)": 0,
                "Tokens": 0
            })
            continue
        examples = form_example(criterion, is_mc, transcript_id) 
        prompt = construct_prompt(transcript, form_row, is_mc, examples)

        try:
            response, duration, tokens, model = call_model(prompt)
            note = extract_masked_note(response)
        except Exception as e:
            response, duration, tokens, model, note = str(e), 0, 0, MODEL_NAME, "ERROR"

        gpt_responses.append(note)

        full_model_log.append({
            "ID": transcript_id,
            "Model Output": response,
            "Model": model,
            "Time Spent (s)": round(duration, 2),
            "Tokens": tokens
        })

    df_gpt[criterion] = gpt_responses

    pd.DataFrame(full_model_log).to_csv(
        os.path.join(MODEL_OUTPUT_DIR, f"{criterion}_output.csv"), index=False)

# ---------------------- FINAL SAVE ----------------------

df_gpt.to_csv(FINAL_ANNOTATION_OUTPUT, index=False)
print(f"\n Annotation complete. File saved to {FINAL_ANNOTATION_OUTPUT}")
