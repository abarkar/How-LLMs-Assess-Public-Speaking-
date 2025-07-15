import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os

# Load datasets
original = pd.read_csv("balanced_subset_original_scores.csv")
expert = pd.read_csv("expert_annotations.csv")
gpt = pd.read_csv("gpt_annotations.csv")

# Settings
dimension_original = 'global'
dimension_expert = 'Global'
dimension_gpt = 'Global'
id_col = 'ID'

# Align by ID
original = original.set_index(id_col)
expert = expert.set_index(id_col)
gpt = gpt.set_index(id_col)

# Ensure common sample order
common_ids = original.index.intersection(expert.index).intersection(gpt.index)
original = original.loc[common_ids]
expert = expert.loc[common_ids]
gpt = gpt.loc[common_ids]

# Create dataframe with annotations and differences
df = pd.DataFrame({
    'ID': common_ids,
    'Original': original[dimension_original],
    'Expert': expert[dimension_expert],
    'GPT': gpt[dimension_gpt],
    'Diff_Expert_vs_GPT': (expert[dimension_expert] - gpt[dimension_gpt]).abs(),
    'Diff_Expert_vs_Original': (expert[dimension_expert] - original[dimension_original]).abs(),
    'Diff_GPT_vs_Original': (gpt[dimension_gpt] - original[dimension_original]).abs(),
    'Prixjury': original['Prixjury'],
    'Prixpublic': original['Prixpublic'],
    'Expert_Comment': expert['Comment'] if 'Comment' in expert.columns else ''
})

# Save to CSV
output_dir = "annotations_interrelations"
os.makedirs(output_dir, exist_ok=True)
df.to_csv(os.path.join(output_dir, "global_pairwise_annotation_differences.csv"), index=False)

# Plot histograms of pairwise differences
plt.figure(figsize=(10, 6))
bins = np.linspace(0, df[['Diff_Expert_vs_GPT', 'Diff_Expert_vs_Original', 'Diff_GPT_vs_Original']].values.max(), 20)

plt.hist(df['Diff_Expert_vs_GPT'], bins=bins, alpha=0.6, label='Expert vs GPT', color='#FF9999', edgecolor='black')
plt.hist(df['Diff_Expert_vs_Original'], bins=bins, alpha=0.6, label='Expert vs Original', color='#99CCFF', edgecolor='black')
plt.hist(df['Diff_GPT_vs_Original'], bins=bins, alpha=0.6, label='GPT vs Original', color='#99FF99', edgecolor='black')

plt.xlabel('Absolute Score Difference')
plt.ylabel('Number of Samples')
plt.title('Pairwise Differences Between Annotations')
plt.legend()
plt.grid(True, linestyle='--', alpha=0.5)
plt.tight_layout()
plt.savefig(os.path.join(output_dir, "global_pairwise_differences_histogram.png"), dpi=300)
plt.show()
