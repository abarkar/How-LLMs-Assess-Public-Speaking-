import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import cohen_kappa_score
from pingouin import intraclass_corr
from itertools import combinations

# Output directory
PROMPTING="few-shot"
output_dir = f"./cross_model_analysis/{PROMPTING}"
os.makedirs(output_dir, exist_ok=True)

# Criteria definitions
option_criteria = [
    "Présentation du Sujet", "Structure", "Niveau de Langue", "Voix Passive",
    "Concision", "Redondance", "Langage Négatif", "Métaphore", "Discours"
]

likert_criteria = [
    "L'introduction", "La conclusion", "Persuasivité", "Clarté du Langage",
    "Créativité du discours", "Global"
]

# Helper function to preprocess model data
def preprocess(df):
    for crit in likert_criteria:
        df[crit] = pd.to_numeric(df[crit], errors='coerce')
    return df

# Load and preprocess all annotations
expert = preprocess(pd.read_csv("expert_annotations.csv"))
gpt4o = preprocess(pd.read_csv(f"gpt-4o_{PROMPTING}_annotations.csv"))
gpt4omini = preprocess(pd.read_csv(f"gpt-4o-mini_{PROMPTING}_annotations.csv"))
gpt41 = preprocess(pd.read_csv(f"gpt-4.1_{PROMPTING}_annotations.csv"))

models = {
    "gpt-4o": gpt4o,
    "gpt-4o-mini": gpt4omini,
    "gpt-4.1": gpt41
}

# Pairwise model agreement
for (name1, df1), (name2, df2) in combinations(models.items(), 2):
    merged = pd.merge(df1, df2, on="ID", suffixes=('_1', '_2'))
    results = []
    for criterion in option_criteria:
        kappa = cohen_kappa_score(merged[f"{criterion}_1"], merged[f"{criterion}_2"])
        results.append([criterion, "Cohen's Kappa", kappa])
    for criterion in likert_criteria:
        icc_data = merged[[f"{criterion}_1", f"{criterion}_2"]].copy()
        icc_data.columns = ["Rater1", "Rater2"]
        icc_data["targets"] = merged["ID"]
        melted = icc_data.melt(id_vars="targets", var_name="rater", value_name="score")
        melted["score"] = pd.to_numeric(melted["score"], errors="coerce")
        try:
            icc_result = intraclass_corr(
                data=melted,
                targets='targets',
                raters='rater',
                ratings='score',
                nan_policy='omit'
            )
            icc_val = icc_result.set_index('Type').loc['ICC2', 'ICC']
        except Exception:
            icc_val = np.nan
        results.append([criterion, "ICC(2,1)", icc_val])
    pd.DataFrame(results, columns=["Criterion", "Metric", "Value"]).to_csv(
        f"{output_dir}/{name1}_vs_{name2}_agreement.csv", index=False
    )

# Expert vs model agreement
expert_agreement = {criterion: {} for criterion in option_criteria + likert_criteria}
for name, df in models.items():
    merged = pd.merge(df, expert, on="ID", suffixes=('_model', '_expert'))
    for criterion in option_criteria:
        kappa = cohen_kappa_score(merged[f"{criterion}_model"], merged[f"{criterion}_expert"])
        expert_agreement[criterion][name] = kappa
    for criterion in likert_criteria:
        icc_data = merged[[f"{criterion}_model", f"{criterion}_expert"]].copy()
        icc_data.columns = ["Rater1", "Rater2"]
        icc_data["targets"] = merged["ID"]
        melted = icc_data.melt(id_vars="targets", var_name="rater", value_name="score")
        melted["score"] = pd.to_numeric(melted["score"], errors="coerce")
        try:
            icc_result = intraclass_corr(
                data=melted,
                targets='targets',
                raters='rater',
                ratings='score',
                nan_policy='omit'
            )
            icc_val = icc_result.set_index('Type').loc['ICC2', 'ICC']
        except Exception:
            icc_val = np.nan
        expert_agreement[criterion][name] = icc_val

# Save expert agreement matrix
agreement_df = pd.DataFrame(expert_agreement).T
agreement_df.to_csv(f"{output_dir}/expert_model_agreement.csv")

# Plotting function
def plot_agreement(df, criteria, title, filename):
    subset = df.loc[criteria].reset_index().melt(id_vars='index', var_name='Model', value_name='Agreement')
    plt.figure(figsize=(12, 6))
    sns.lineplot(data=subset, x='index', y='Agreement', hue='Model', marker='o')
    plt.xticks(rotation=45, ha='right')
    plt.title(title)
    plt.ylabel("Agreement")
    plt.xlabel("Criterion")
    plt.tight_layout()
    plt.savefig(f"{output_dir}/{filename}")
    plt.close()

# Generate plots
plot_agreement(agreement_df, option_criteria, "Expert vs Models (Categorical Criteria)", "expert_vs_models_option.png")
plot_agreement(agreement_df, likert_criteria, "Expert vs Models (Likert-Scaled Criteria)", "expert_vs_models_likert.png")
