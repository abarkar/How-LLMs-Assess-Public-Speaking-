import os
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import chi2_contingency, kruskal, zscore
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from mpl_toolkits.mplot3d import Axes3D


annotation_type = "gpt-4o-mini_zero-shot"

# --- Setup ---
output_dir = f'./analysis_of_{annotation_type}_annotations'
os.makedirs(output_dir, exist_ok=True)

# Load data
df = pd.read_csv(f"{annotation_type}_annotations.csv")

# Define column groups
id_col = df.columns[0]
mc_cols = df.columns[1:10]
likert_cols = df.columns[10:15]
hundred_point_col = df.columns[15]
time_col = 'Annotation Time'

import scipy.stats as stats
import matplotlib.colors as mcolors

# -------- STEP 1: Prepare all criteria for unified Spearman correlation --------
df_corr = df.copy()

# Custom recoding for specific MC criteria (binary logic)
custom_map = {
    "Concision": {"A": 0, "B": 1, "C": 0},
    "Redondance": {"A": 0, "B": 1, "C": 0},
    "Voix Passive": {"A": 0, "B": 1, "C": 0},
    "Langage Négatif": {"A": 0, "B": 1, "C": 0},
}
for col, mapping in custom_map.items():
    df_corr[col] = df_corr[col].map(mapping)

# Default ordinal encoding for remaining MC criteria (A < B < C)
for col in mc_cols:
    if col not in custom_map:
        df_corr[col] = df_corr[col].map({"A": 0, "B": 1, "C": 2})

# Include Likert and global score
all_criteria_for_corr = list(mc_cols) + list(likert_cols) + [hundred_point_col]

# -------- STEP 2: Spearman Correlation Matrix with P-values --------
rho_matrix = pd.DataFrame(index=all_criteria_for_corr, columns=all_criteria_for_corr, dtype=float)
pval_matrix = pd.DataFrame(index=all_criteria_for_corr, columns=all_criteria_for_corr, dtype=float)

for col1 in all_criteria_for_corr:
    for col2 in all_criteria_for_corr:
        rho, pval = stats.spearmanr(df_corr[col1], df_corr[col2])
        rho_matrix.loc[col1, col2] = rho
        pval_matrix.loc[col1, col2] = pval

# Save results to CSV
rho_matrix.to_csv(f"{output_dir}/full_spearman_correlation_matrix.csv")
pval_matrix.to_csv(f"{output_dir}/full_spearman_pvalues.csv")


ordered_criteria = [
    "Présentation du Sujet", "Structure", "Niveau de Langue", "Voix Passive", "Concision",
    "Redondance", "Langage Négatif", "Métaphore", "Discours", "L'introduction", "La conclusion",
    "Persuasivité", "Clarté du Langage", "Créativité du discours", "Global"
]

# -------- STEP 3: Heatmap with groups, English labels, and lower triangle --------

# Translation dictionary
rename_dict = {
    "Présentation du Sujet": "Topic Presentation",
    "Structure": "Structure",
    "Niveau de Langue": "Language Level",
    "Voix Passive": "Passive Voice",
    "Concision": "Conciseness",
    "Redondance": "Redundancy",
    "Langage Négatif": "Negative Language",
    "Métaphore": "Metaphor",
    "Discours": "Storytelling",
    "L'introduction": "Introduction",
    "La conclusion": "Conclusion",
    "Persuasivité": "Parsuasiveness",
    "Clarté du Langage": "Clarity of Language",
    "Créativité du discours": "Creativity of Speech",
    "Global": "Global"
}

# Define groupings
mc_criteria = [
    "Présentation du Sujet", "Structure", "Niveau de Langue", "Voix Passive", "Concision",
    "Redondance", "Langage Négatif", "Métaphore", "Discours"
]
sc_criteria = ["L'introduction", "La conclusion"]
sd_dimensions = ["Persuasivité", "Clarté du Langage", "Créativité du discours", "Global"]

# Final display order in English
ordered_criteria = mc_criteria + sc_criteria + sd_dimensions
ordered_criteria_en = [rename_dict[col] for col in ordered_criteria]

# Rename matrices
rho_renamed = rho_matrix.rename(index=rename_dict, columns=rename_dict)
pval_renamed = pval_matrix.rename(index=rename_dict, columns=rename_dict)

# Reorder
rho_renamed = rho_renamed.loc[ordered_criteria_en, ordered_criteria_en]
pval_renamed = pval_renamed.loc[ordered_criteria_en, ordered_criteria_en]

# Create masking
mask_upper = np.triu(np.ones_like(rho_renamed, dtype=bool))
mask_nonsig = pval_renamed.astype(float) >= 0.05
full_mask = mask_upper | mask_nonsig

# Determine group boundaries
mc_end = len(mc_criteria)
sc_end = mc_end + len(sc_criteria)

# Plot
plt.figure(figsize=(12, 10))
ax = sns.heatmap(
    rho_renamed.astype(float),
    annot=rho_renamed.where(~full_mask).round(2),
    fmt='',
    mask=full_mask,
    cmap='coolwarm',
    center=0,
    square=True,
    cbar_kws={"label": "Spearman Correlation"}
)

# Add horizontal/vertical lines to separate groups
for idx in [mc_end, sc_end]:
    ax.axhline(idx, color='black', linewidth=2)
    ax.axvline(idx, color='black', linewidth=2)

# Formatting
plt.title("Spearman Correlation Between All Criteria\n(Significant, Lower Triangle Only)")
plt.xticks(rotation=45, ha='right')
plt.yticks(rotation=0)
plt.tight_layout()
plt.savefig(f"{output_dir}/spearman_heatmap_with_groups.png")
plt.close()



import os
import pandas as pd
import numpy as np
from scipy.stats import spearmanr
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

# Setup paths
features_dir = '../features'
lexical_corr_dir = os.path.join(output_dir, 'lexical_correlations')
os.makedirs(lexical_corr_dir, exist_ok=True)

# Prepare correctly ranked criteria dataframe
df_ranked = df.copy()
custom_map = {
    "Concision": {"A": 0, "B": 1, "C": 0},
    "Redondance": {"A": 0, "B": 1, "C": 0},
    "Voix Passive": {"A": 0, "B": 1, "C": 0},
    "Langage Négatif": {"A": 0, "B": 1, "C": 0},
}
for col, mapping in custom_map.items():
    df_ranked[col] = df_ranked[col].map(mapping)
for col in mc_cols:
    if col not in custom_map:
        df_ranked[col] = df_ranked[col].map({"A": 0, "B": 1, "C": 2})
# criteria_df = df_ranked.set_index(id_col)[mc_cols + list(likert_cols) + [hundred_point_col]]
criteria_df = df_ranked.set_index(id_col)[list(mc_cols) + list(likert_cols) + [hundred_point_col]]
# criteria_df.columns = criteria_df.columns.str.replace(' ', '_')

# STEP 1-2: Correlation between lexical features and criteria
summary_stats = []

for fname in os.listdir(features_dir):
    print(fname)
    if fname.endswith('.csv'):
        category = fname.replace('.csv', '')
        lex_df = pd.read_csv(os.path.join(features_dir, fname)).set_index(id_col)
        # merged = pd.merge(lex_df, criteria_df, left_index=True, right_index=True)
        merged = pd.merge(lex_df, criteria_df, on="ID")
        print(len(lex_df))
        print(len(merged))
        # Compute Spearman correlation
        records = []
        for feature in lex_df.columns:
            for crit in criteria_df.columns:
                coef, pval = spearmanr(merged[feature], merged[crit])
                records.append({
                    "Lexical_Feature": feature,
                    "Criterion": crit,
                    "Spearman_Coefficient": coef,
                    "p_value": pval
                })
        df_result = pd.DataFrame(records)
        df_result.to_csv(f"{lexical_corr_dir}/{category}_spearman.csv", index=False)
        
        # Filter only significant correlations
        df_sig = df_result[df_result["p_value"] < 0.05]

        # Aggregate summary on significant values only
        summary = df_sig.groupby("Criterion")["Spearman_Coefficient"].agg(['mean', 'min', 'max']).reset_index()
        summary['Category'] = category
        summary_stats.append(summary)



# # Define the fixed order
# ordered_criteria = [
#     "Présentation du Sujet", "Structure", "Niveau de Langue", "Voix Passive", "Concision", "Redondance",
#     "Langage Négatif", "Métaphore", "Discours", "L'introduction", "La conclusion",
#     "Persuasivité", "Clarté du Langage", "Créativité du discours", "Global"
# ]
# # STEP 3: Visualization of aggregated correlations with fixed criterion order
# summary_df = pd.concat(summary_stats)
# summary_df['Criterion'] = pd.Categorical(summary_df['Criterion'], categories=ordered_criteria, ordered=True)
# summary_df = summary_df.sort_values('Criterion')

# plt.figure(figsize=(14, 6))

# # Assign fixed colors by category name
# category_colors = {
#     'lexical_diversity': '#1f77b4',
#     'negation': '#ff7f0e',
#     'overlap': '#2ca02c',
#     'passive_voice': '#d62728',
#     'pos_tag_density': '#9467bd',
#     'storytelling': '#8c564b',
#     'syllable_readability_metrics': '#e377c2',
#     'syntactic_diversity': '#16e7ce',
#     'transitions': '#7f7f7f',

# }
# # Ensure categorical ordering globally
# summary_df['Criterion'] = pd.Categorical(summary_df['Criterion'], categories=ordered_criteria, ordered=True)

# plt.figure(figsize=(14, 6))

# for cat in summary_df['Category'].unique():
#     subset = summary_df[summary_df['Category'] == cat].set_index('Criterion').reindex(ordered_criteria)
    
#     # Plot with NaN-aware errorbar (NaNs will cause breaks in the lines)
#     plt.errorbar(
#         ordered_criteria,  # x-axis is always full and ordered
#         subset['mean'],
#         yerr=[subset['mean'] - subset['min'], subset['max'] - subset['mean']],
#         label=cat,
#         fmt='o-', capsize=4,
#         color=category_colors.get(cat, None)
#     )

# plt.axhline(0, linestyle='--', color='gray')
# plt.xticks(rotation=45, ha='right')
# plt.title("Spearman Correlation Summary: Lexical Categories vs Criteria\n(Significant Only)")
# plt.ylabel("Mean Spearman Correlation ± Range")
# plt.legend(title="Lexical Category", bbox_to_anchor=(1.05, 1), loc='upper left')
# plt.tight_layout()
# plt.savefig(f"{output_dir}/{annotation_type}_lexical_category_vs_criteria_summary_significant.png")
# plt.close()



# # STEP 4: PCA with lexical + MC features
# all_lexical = []
# for fname in os.listdir(features_dir):
#     if fname.endswith('.csv'):
#         all_lexical.append(pd.read_csv(os.path.join(features_dir, fname)).set_index(id_col))
# lexical_full = pd.concat(all_lexical, axis=1)
# print(criteria_df.columns)
# joint_df = pd.merge(lexical_full, criteria_df[mc_cols], left_index=True, right_index=True)

# X = joint_df.copy()
# X_scaled = StandardScaler().fit_transform(X)

# pca_model = PCA(n_components=3)
# pca_coords = pca_model.fit_transform(X_scaled)
# loadings = pd.DataFrame(pca_model.components_.T, index=X.columns, columns=["PC1", "PC2", "PC3"])
# loadings.to_csv(f"{output_dir}/pca_lexical_mc_loadings.csv")

# # Check which columns are valid before plotting
# valid_criteria = [col for col in list(likert_cols) + [hundred_point_col] if col in df_ranked.columns]
# print("Expected criteria columns:", list(likert_cols) + [hundred_point_col])
# print("Available columns in df_ranked:", df_ranked.columns.tolist())

# for crit in valid_criteria:
#     fig = plt.figure(figsize=(10, 8))
#     ax = fig.add_subplot(111, projection='3d')
#     sc = ax.scatter(
#         pca_coords[:, 0], pca_coords[:, 1], pca_coords[:, 2],
#         c=df_ranked[crit], cmap='viridis', s=60
#     )
#     fig.colorbar(sc, ax=ax, shrink=0.6).set_label(crit)
#     ax.set_title(f"PCA (Lexical + MC) — Colored by {crit}")
#     ax.set_xlabel(f"PC1 ({pca_model.explained_variance_ratio_[0]*100:.1f}%)")
#     ax.set_ylabel(f"PC2 ({pca_model.explained_variance_ratio_[1]*100:.1f}%)")
#     ax.set_zlabel(f"PC3 ({pca_model.explained_variance_ratio_[2]*100:.1f}%)")
#     plt.tight_layout()
#     plt.savefig(f"{output_dir}/pca_lexical_mc_by_{crit}.png")
#     plt.close()


# # Correlation Circle
# fig = plt.figure(figsize=(10, 8))
# ax = fig.add_subplot(111, projection='3d')
# for i, var in enumerate(X.columns):
#     ax.plot([0, loadings.loc[var, "PC1"]], [0, loadings.loc[var, "PC2"]], [0, loadings.loc[var, "PC3"]],
#             color='gray', alpha=0.5)
#     ax.text(loadings.loc[var, "PC1"] * 1.3, loadings.loc[var, "PC2"] * 1.3, loadings.loc[var, "PC3"] * 1.3,
#             var, fontsize=7)
# ax.set_xlim([-1, 1])
# ax.set_ylim([-1, 1])
# ax.set_zlim([-1, 1])
# ax.set_xlabel("PC1")
# ax.set_ylabel("PC2")
# ax.set_zlabel("PC3")
# ax.set_title("PCA Correlation Circle (Lexical + MC)")
# plt.tight_layout()
# plt.savefig(f"{output_dir}/pca_lexical_mc_correlation_circle.png")
# plt.close()
