import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# Load data
original = pd.read_csv("balanced_subset_original_scores.csv")
expert = pd.read_csv("expert_annotations.csv")
gpt = pd.read_csv("gpt_annotations.csv")

# Settings
dimension = 'Persuasivité'  # <-- easily change here if needed
original_col = 'persuasiveness'  # column name in original file
id_col = 'ID'

# Sort by original persuasiveness
sorted_ids = original.sort_values(original_col)[id_col].tolist()

# Merge information
original = original.set_index(id_col)
expert = expert.set_index(id_col)
gpt = gpt.set_index(id_col)

# Create aligned lists
x = np.arange(len(sorted_ids))
original_scores = original.loc[sorted_ids, original_col]
expert_scores = expert.loc[sorted_ids, dimension]
gpt_scores = gpt.loc[sorted_ids, dimension]

# Global font size settings
plt.rcParams.update({
    'font.size': 25,
    'axes.titlesize': 25,
    'axes.labelsize': 25,
    'xtick.labelsize': 25,
    'ytick.labelsize': 25,
    'legend.fontsize': 25,
})

# Create figure
fig, ax = plt.subplots(figsize=(20, 8))

# Plot the three trends
ax.plot(x, original_scores, label='Original (Harmonic Mean)', color='black', linestyle='-', marker='o', markersize=3)
ax.plot(x, expert_scores, label='Expert Annotation', color='red', linestyle='--', marker='^', markersize=3)
ax.plot(x, gpt_scores, label='GPT Annotation', color='purple', linestyle=':', marker='s', markersize=3)

# Mark jury and public prizes
for idx, sample_id in enumerate(sorted_ids):
    if original.loc[sample_id, 'Prixjury'] in [1, 2, 3]:
        ax.axvline(x=idx, color='green', linestyle='--', alpha=0.5, linewidth=1)
    if original.loc[sample_id, 'Prixpublic'] == 1:
        ax.axvline(x=idx, color='blue', linestyle='--', alpha=0.5, linewidth=1)

# Mark comments from expert
for idx, sample_id in enumerate(sorted_ids):
    comment = expert.loc[sample_id, 'Comment'] if 'Comment' in expert.columns else ''
    if pd.notnull(comment) and str(comment).strip() != '':
        ax.plot(idx, 0.83, marker='*', markersize=8, color='orange', clip_on=False)

# X-axis ticks with readable sample IDs
ax.set_xticks(x)
ax.set_xticklabels(sorted_ids, rotation=45, fontsize=12)

# Labels and title
ax.set_xlabel('Sample ID (sorted by original persuasiveness)')
ax.set_ylabel('Persuasiveness Score')
ax.set_title(f'{dimension} Trends Across Samples')

# Y-axis limits
ax.set_ylim(0.8, 5.3)

# Legend (choose placement)
ax.legend(loc='upper left')
# ax.legend(loc='center left', bbox_to_anchor=(1.02, 0.5))  # To move it outside

# Save and show
plt.tight_layout()
plt.savefig(f"annotations_interrelations/trend_plot_{dimension.lower()}.png", dpi=300)
plt.show()
