import os
import pandas as pd
import numpy as np
from sklearn.metrics import cohen_kappa_score
from scipy.stats import spearmanr, pointbiserialr

# Setup
output_dir = "./annotations_interrelations"
os.makedirs(output_dir, exist_ok=True)

# Load data
expert = pd.read_csv("expert_annotations.csv")
gpt = pd.read_csv("gpt_annotations.csv")
original = pd.read_csv("balanced_subset_original_scores.csv")

# Sort and align
expert = expert.sort_values('ID').reset_index(drop=True)
gpt = gpt.sort_values('ID').reset_index(drop=True)
original = original.sort_values('ID').reset_index(drop=True)

# Recode MC
mc_cols = expert.columns[1:10]
custom_map = {
    "Concision": {"A": 0, "B": 1, "C": 0},
    "Redondance": {"A": 0, "B": 1, "C": 0},
    "Voix Passive": {"A": 0, "B": 1, "C": 0},
    "Langage Négatif": {"A": 0, "B": 1, "C": 0},
}
for df in [expert, gpt]:
    for col, mapping in custom_map.items():
        df[col] = df[col].map(mapping)
    for col in mc_cols:
        if col not in custom_map:
            df[col] = df[col].map({"A": 0, "B": 1, "C": 2})

# Fix prizes
original["Prixjury"] = original["Prixjury"].notnull().astype(int)
original["Prixpublic"] = original["Prixpublic"].fillna(0).astype(int)

# Define Likert columns
likert_cols = expert.columns[10:16].tolist()
original_scores = ['selfConfidence', 'persuasiveness', 'engagement', 'global']
rater_cols = ['persuasiveness_rater1', 'persuasiveness_rater2', 'persuasiveness_rater3']

# --- 1. Cohen’s Kappa for MC Criteria ---
kappa_scores = {col: cohen_kappa_score(expert[col], gpt[col]) for col in mc_cols}
pd.Series(kappa_scores, name="Cohen_Kappa").to_csv(f"{output_dir}/cohen_kappa_scores.csv")

# --- 2. ICC manually (two-way random average agreement ICC(2,1)) ---
def compute_icc(data1, data2):
    df = pd.DataFrame({"r1": data1, "r2": data2})
    df = df.dropna()
    n = len(df)
    m = 2
    MSR = np.var(df.mean(axis=1), ddof=1) * m
    MSE = np.mean((df["r1"] - df["r2"])**2) / 2
    ICC = (MSR - MSE) / (MSR + (m - 1) * MSE)
    return ICC

icc_results = [(col, compute_icc(expert[col], gpt[col])) for col in likert_cols]
pd.DataFrame(icc_results, columns=["Criterion", "ICC2"]).to_csv(f"{output_dir}/icc_expert_vs_gpt.csv", index=False)

# --- 3. Persuasiveness Correlation with Individual & Harmonic Raters ---
def correlate_with_raters(source, label):
    results = []
    for r in rater_cols:
        corr, p = spearmanr(source["Persuasivité"], original[r])
        results.append((r, corr, p))
    # Harmonic mean
    raters_matrix = original[rater_cols].replace(0, np.nan)
    harmonic = 3 / np.nansum(1.0 / raters_matrix, axis=1)
    corr, p = spearmanr(source["Persuasivité"], harmonic)
    results.append(("harmonic_mean", corr, p))
    pd.DataFrame(results, columns=["Rater", "Spearman_Corr", "p-value"]).to_csv(
        f"{output_dir}/correlation_persuasiveness_{label}.csv", index=False)

correlate_with_raters(expert, "expert")
correlate_with_raters(gpt, "gpt")

# --- 4. Cross-correlation with full Likert ---
def save_spearman_matrix(base_df, ann_df, base_names, ann_names, name):
    rows = []
    for base in base_names:
        for ann in ann_names:
            corr, p = spearmanr(base_df[base], ann_df[ann])
            rows.append((base, ann, corr, p))
    df = pd.DataFrame(rows, columns=["Original", "Annotation", "Spearman", "p-value"])
    df.to_csv(f"{output_dir}/cross_corr_{name}.csv", index=False)

save_spearman_matrix(original, expert, original_scores, likert_cols, "expert")
save_spearman_matrix(original, gpt, original_scores, likert_cols, "gpt")

# --- 5. Correlation with Jury/Public Prizes ---
from scipy.stats import mannwhitneyu

def correlate_with_prizes_ordinal(ann_df, name):
    spearman_results = []
    mannwhitney_results = []

    for prize in ['Prixjury', 'Prixpublic']:
        for col in mc_cols.tolist() + likert_cols:
            if col in ann_df.columns:
                # Spearman correlation
                corr, pval = spearmanr(original[prize], ann_df[col])
                spearman_results.append((prize, col, corr, pval))

                # Mann-Whitney U test
                try:
                    group0 = ann_df[original[prize] == 0][col]
                    group1 = ann_df[original[prize] == 1][col]
                    u_stat, u_p = mannwhitneyu(group0, group1, alternative='two-sided')
                    mannwhitney_results.append((prize, col, u_stat, u_p))
                except Exception as e:
                    mannwhitney_results.append((prize, col, np.nan, f'Error: {e}'))

    pd.DataFrame(spearman_results, columns=["Prize", "Annotation", "Spearman_Corr", "p-value"])\
        .to_csv(f"{output_dir}/prize_spearman_corr_{name}.csv", index=False)
    
    pd.DataFrame(mannwhitney_results, columns=["Prize", "Annotation", "U_stat", "p-value"])\
        .to_csv(f"{output_dir}/prize_mannwhitney_{name}.csv", index=False)

# Apply to expert and gpt annotations
correlate_with_prizes_ordinal(expert, "expert")
correlate_with_prizes_ordinal(gpt, "gpt")
