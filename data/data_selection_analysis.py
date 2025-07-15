import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import kstest, normaltest, uniform
import os

# 1. Load original dataset and filter
original_df = pd.read_csv("MT_aggregated_ratings.csv")
filtered_original = original_df[
    (original_df['clip'] == 'full') & 
    (original_df['aggregationMethod'] == 'harmmean')
][['ID', 'persuasiveness', 'global']].copy()

# 2. Load balanced dataset
balanced_df = pd.read_csv("balanced_subset_original_scores.csv")[['ID', 'persuasiveness', 'global']].copy()

# Ensure output directory exists
output_dir = "distribution_analysis_results"
os.makedirs(output_dir, exist_ok=True)

# Function to compute and save distribution statistics and test results
def analyze_and_save(data, label, filename):
    desc = data.describe()
    normal_stat, normal_p = normaltest(data)
    ks_stat, ks_p = kstest(data, 'uniform', args=(data.min(), data.max() - data.min()))

    with open(os.path.join(output_dir, filename), "w") as f:
        f.write(f"--- {label} ---\n")
        f.write("Descriptive Statistics:\n")
        f.write(f"{desc.to_string()}\n\n")
        f.write("Normality Test (D'Agostino and Pearson):\n")
        f.write(f"Statistic={normal_stat:.4f}, p-value={normal_p:.4f}\n\n")
        f.write("Uniformity Test (Kolmogorov–Smirnov):\n")
        f.write(f"Statistic={ks_stat:.4f}, p-value={ks_p:.4f}\n")

# Analyze and save all distributions
analyze_and_save(filtered_original['persuasiveness'], "Original Persuasiveness", "original_persuasiveness.txt")
analyze_and_save(filtered_original['global'], "Original Global", "original_global.txt")
analyze_and_save(balanced_df['persuasiveness'], "Balanced Persuasiveness", "balanced_persuasiveness.txt")
analyze_and_save(balanced_df['global'], "Balanced Global", "balanced_global.txt")

# 3. Plot histograms with fixed bins
def plot_histogram(original_data, balanced_data, title, bins, color1="#FDB88D", color2="#871A31"):
    plt.figure(figsize=(8, 5))
    plt.hist(original_data, bins=bins, alpha=0.6, label='Original', color=color1, edgecolor='black', density=False)
    plt.hist(balanced_data, bins=bins, alpha=0.6, label='Balanced', color=color2, edgecolor='black', density=False)
    plt.xlabel('Score')
    plt.ylabel('Count')
    plt.title(f'Distribution of {title}')
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.5)
    plt.tight_layout()
    plt.savefig(f"./distribution_analysis_results/{title}.png")
    plt.show()

# Define bins for each score
persuasiveness_bins = np.arange(0, 6, 1)  # 0-1, 1-2, ..., 4-5
global_bins = np.arange(0, 110, 10)       # 0-10, ..., 90-100

# Plot histograms
plot_histogram(filtered_original['persuasiveness'], balanced_df['persuasiveness'], "Persuasiveness", persuasiveness_bins)
plot_histogram(filtered_original['global'], balanced_df['global'], "Global", global_bins)



# Define column types
categorical_columns = [
    "Présentation du Sujet", "Structure", "Niveau de Langue", "Voix Passive",
    "Concision", "Redondance", "Langage Négatif", "Métaphore", "Discours"
]

likert_columns = [
    "L'introduction", "La conclusion", "Persuasivité",
    "Clarté du Langage", "Créativité du discours", "Global"
]

# Load expert annotations
expert_df = pd.read_csv("expert_annotations.csv")
gpt_df = pd.read_csv("gpt_annotations.csv")

# Output directory for expert analysis
expert_output_dir = os.path.join(output_dir, "expert")
os.makedirs(expert_output_dir, exist_ok=True)
gpt_output_dir = os.path.join(output_dir, "gpt")
os.makedirs(gpt_output_dir, exist_ok=True)

# Analyze categorical columns
for col in categorical_columns:
    counts = expert_df[col].value_counts()
    with open(os.path.join(expert_output_dir, f"{col.replace(' ', '_')}_stats.txt"), "w") as f:
        f.write(f"--- {col} (Categorical A/B/C) ---\n")
        f.write("Counts:\n")
        f.write(f"{counts.to_string()}\n")
    counts = gpt_df[col].value_counts()
    with open(os.path.join(gpt_output_dir, f"{col.replace(' ', '_')}_stats.txt"), "w") as f:
        f.write(f"--- {col} (Categorical A/B/C) ---\n")
        f.write("Counts:\n")
        f.write(f"{counts.to_string()}\n")

# Analyze Likert scale columns
def distribution_type(df, folder):
    for col in likert_columns:
        desc = df[col].describe()
        normal_stat, normal_p = normaltest(df[col])
        ks_stat, ks_p = kstest(df[col], 'uniform', args=(df[col].min(), df[col].max() - df[col].min()))

        with open(os.path.join(folder, f"{col.replace(' ', '_')}_stats.txt"), "w") as f:
            f.write(f"--- {col} (Likert Scale) ---\n")
            f.write("Descriptive Statistics:\n")
            f.write(f"{desc.to_string()}\n\n")
            f.write("Normality Test (D'Agostino and Pearson):\n")
            f.write(f"Statistic={normal_stat:.4f}, p-value={normal_p:.4f}\n\n")
            f.write("Uniformity Test (Kolmogorov–Smirnov):\n")
            f.write(f"Statistic={ks_stat:.4f}, p-value={ks_p:.4f}\n")

distribution_type(expert_df, expert_output_dir)
distribution_type(gpt_df, gpt_output_dir)


# Plot 3-way histograms for Persuasiveness and Global
def plot_three_way_histogram(original_data, balanced_data, expert_data, title, bins, colors, labels):
    plt.figure(figsize=(8, 5))
    plt.hist(original_data, bins=bins, alpha=0.5, label=labels[0], color=colors[0], edgecolor='black')
    plt.hist(balanced_data, bins=bins, alpha=0.5, label=labels[1], color=colors[1], edgecolor='black')
    plt.hist(expert_data, bins=bins, alpha=0.5, label=labels[2], color=colors[2], edgecolor='black')
    plt.xlabel('Score')
    plt.ylabel('Count')
    plt.title(f'Distribution of {title}')
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.5)
    plt.tight_layout()
    plt.savefig(f"{output_dir}/Expert_{title}.png")
    plt.show()

# Plot Persuasiveness
plot_three_way_histogram(
    original_data=filtered_original['persuasiveness'],
    balanced_data=balanced_df['persuasiveness'],
    expert_data=expert_df['Persuasivité'],
    title="Persuasiveness",
    bins=np.arange(0, 6, 1),
    colors=["#FDB88D", "#871A31", "#5171A5"],
    labels=["Original", "Balanced", "Expert"]
)



# Plot Global
plot_three_way_histogram(
    original_data=filtered_original['global'],
    balanced_data=balanced_df['global'],
    expert_data=expert_df['Global'],
    title="Global",
    bins=np.arange(0, 110, 10),
    colors=["#FDB88D", "#871A31", "#5171A5"],
    labels=["Original", "Balanced", "Expert"]
)
