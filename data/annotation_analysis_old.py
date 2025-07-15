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

# --- Setup ---
output_dir = './analysis_of_expert_annotations'
os.makedirs(output_dir, exist_ok=True)

# Load data
df = pd.read_csv("expert_annotations.csv")

# Define column groups
id_col = df.columns[0]
mc_cols = df.columns[1:10]
likert_cols = df.columns[10:15]
hundred_point_col = df.columns[15]
time_col = 'Annotation Time'

# -------- STEP 1: Distributions of Multiple-Choice --------
for col in mc_cols:
    plt.figure()
    sns.countplot(data=df, x=col, order=["A", "B", "C"])
    plt.title(f'Distribution of responses for {col}')
    plt.savefig(f"{output_dir}/dist_{col}.png")
    plt.close()

# -------- STEP 2: Cramér’s V Heatmap for MC ----------
def cramers_v(x, y):
    confusion_matrix = pd.crosstab(x, y)
    chi2 = chi2_contingency(confusion_matrix)[0]
    n = confusion_matrix.sum().sum()
    phi2 = chi2 / n
    r, k = confusion_matrix.shape
    phi2_corr = max(0, phi2 - ((k - 1) * (r - 1)) / (n - 1))
    r_corr = r - ((r - 1) ** 2) / (n - 1)
    k_corr = k - ((k - 1) ** 2) / (n - 1)
    return np.sqrt(phi2_corr / min((k_corr - 1), (r_corr - 1)))

cramers_results = pd.DataFrame(index=mc_cols, columns=mc_cols)
for col1 in mc_cols:
    for col2 in mc_cols:
        cramers_results.loc[col1, col2] = cramers_v(df[col1], df[col2])
cramers_results = cramers_results.astype(float)
cramers_results.to_csv(f"{output_dir}/cramers_v_matrix.csv")

plt.figure(figsize=(10, 8))
sns.heatmap(cramers_results, annot=True, cmap="coolwarm", square=True)
plt.title("Cramér’s V between Multiple-Choice Criteria")
plt.xticks(rotation=45, ha='right')  # Rotate x-axis tick labels
plt.yticks(rotation=0)               # Keep y-axis labels horizontal
plt.tight_layout()
plt.savefig(f"{output_dir}/cramers_v_heatmap.png")
plt.show()

plt.close()

# -------- STEP 3: Likert Correlation Matrix and PCA + Correlation Circle ----------
likert_corr = df[likert_cols].corr()
likert_corr.to_csv(f"{output_dir}/likert_correlation.csv")

plt.figure(figsize=(8, 6))
sns.heatmap(likert_corr, annot=True, cmap="viridis")
plt.title("Correlation Matrix of Likert-Scale Criteria")
plt.xticks(rotation=45, ha='right')  # Horizontal-ish x-axis
plt.yticks(rotation=0)               # Horizontal y-axis
plt.tight_layout()
plt.savefig(f"{output_dir}/likert_corr_heatmap.png")
plt.show()

plt.close()

# PCA for Likert variables
scaler = StandardScaler()
likert_scaled = scaler.fit_transform(df[likert_cols])
pca = PCA(n_components=2)
components = pca.fit_transform(likert_scaled)

plt.figure()
plt.scatter(components[:, 0], components[:, 1])
plt.title("PCA of Likert-Scale Criteria")
plt.xlabel("PC1")
plt.ylabel("PC2")
plt.savefig(f"{output_dir}/pca_likert.png")
plt.show()
plt.close()


# PCA correlation circle
plt.figure(figsize=(6, 6))
for i, var in enumerate(likert_cols):
    plt.arrow(0, 0, pca.components_[0, i], pca.components_[1, i],
              head_width=0.03, color='red', alpha=0.6)
    plt.text(pca.components_[0, i] * 1.1, pca.components_[1, i] * 1.1, var,
             color='black', ha='center', va='center')
plt.xlim(-1, 1)
plt.ylim(-1, 1)
plt.xlabel("PC1")
plt.ylabel("PC2")
plt.grid()
plt.title("PCA Correlation Circle")
plt.savefig(f"{output_dir}/pca_correlation_circle.png")
plt.close()

# -------- STEP 4: MC vs Likert (Boxplots + Kruskal-Wallis) ----------
kruskal_results = []

for mc in mc_cols:
    for likert in likert_cols:
        plt.figure()
        sns.boxplot(x=df[mc], y=df[likert], order=["A", "B", "C"])
        plt.title(f'{likert} by {mc}')
        plt.savefig(f"{output_dir}/boxplot_{likert}_by_{mc}.png")
        plt.close()

        groups = [group[likert].values for name, group in df.groupby(mc)]
        stat, p = kruskal(*groups)
        kruskal_results.append({
            'MC_Criterion': mc,
            'Likert_Criterion': likert,
            'Kruskal_Stat': stat,
            'p_value': p
        })

kruskal_df = pd.DataFrame(kruskal_results)
kruskal_df.to_csv(f"{output_dir}/kruskal_mc_vs_likert.csv", index=False)

# -------- STEP 5: Heatmap of All Annotations Per Sample ----------
df_encoded = df.copy()
le = LabelEncoder()
for col in mc_cols:
    df_encoded[col] = le.fit_transform(df[col])

all_criteria = list(mc_cols) + list(likert_cols) + [hundred_point_col]
data_matrix = df_encoded[all_criteria]

plt.figure(figsize=(12, 6))
sns.heatmap(data_matrix, cmap='coolwarm', cbar=True)
plt.title("Heatmap of All Annotations Per Sample")
plt.xlabel("Criteria")
plt.ylabel("Sample Index")
plt.savefig(f"{output_dir}/sample_annotation_heatmap.png")
plt.close()

# -------- STEP 6: Clustering (original features) + PCA-based clustering ----------
all_criteria = list(mc_cols) + list(likert_cols)

features = df_encoded[all_criteria]
scaled_features = scaler.fit_transform(features)

# Silhouette scores for original feature clustering
silhouette_scores = []
k_range = range(2, 8)
for k in k_range:
    kmeans = KMeans(n_clusters=k, random_state=42)
    labels = kmeans.fit_predict(scaled_features)
    silhouette_scores.append(silhouette_score(scaled_features, labels))

plt.figure()
plt.plot(k_range, silhouette_scores, marker='o')
plt.title('Silhouette Scores for K-Means Clustering')
plt.xlabel('Number of clusters')
plt.ylabel('Silhouette Score')
plt.savefig(f"{output_dir}/silhouette_scores.png")
plt.close()

best_k = np.argmax(silhouette_scores) + 2
kmeans = KMeans(n_clusters=best_k, random_state=42)
cluster_labels = kmeans.fit_predict(scaled_features)
components = PCA(n_components=2).fit_transform(scaled_features)

plt.figure(figsize=(8, 6))
sns.scatterplot(x=components[:, 0], y=components[:, 1], hue=cluster_labels, palette='Set2')
plt.title(f'K-Means Clusters (k={best_k}) on PCA projection')
plt.xlabel('PC1')
plt.ylabel('PC2')
plt.legend(title='Cluster')
plt.savefig(f"{output_dir}/cluster_pca.png")
plt.close()


# PCA before clustering
pca3d_model = PCA(n_components=3)
pca3d = pca3d_model.fit_transform(scaled_features)

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
scatter = ax.scatter(
    pca3d[:, 0], pca3d[:, 1], pca3d[:, 2],
    c=df[hundred_point_col], cmap='viridis', marker='o'
)
cbar = fig.colorbar(scatter, ax=ax, label='Global Score')
ax.set_xlabel(f"PC1 ({pca3d_model.explained_variance_ratio_[0]*100:.1f}%)")
ax.set_ylabel(f"PC2 ({pca3d_model.explained_variance_ratio_[1]*100:.1f}%)")
ax.set_zlabel(f"PC3 ({pca3d_model.explained_variance_ratio_[2]*100:.1f}%)")
ax.set_title("3D PCA with Global Score Coloring")
plt.savefig(f"{output_dir}/pca3d_colored_by_score.png")
plt.show()

plt.close()

from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import matplotlib.colors as mcolors

# Clustering on PCA coordinates
pca_coords = pca3d_model.transform(scaled_features)
kmeans_pca = KMeans(n_clusters=best_k, random_state=42, n_init='auto')
kmeans_labels = kmeans_pca.fit_predict(pca_coords)

# Global score values for color
scores = df[hundred_point_col].values
norm = mcolors.Normalize(vmin=scores.min(), vmax=scores.max())
cmap = cm.viridis
# colors = cmap(norm(scores))
# Adjust colormap brightness
brightness_factor = 0.6  # <1 for brighter, >1 for darker
colors = cmap(norm(scores) ** brightness_factor)


# Define shapes for each cluster
shapes = ['o', '^', 's', 'P', 'D', 'X']
fig = plt.figure(figsize=(10, 8))
ax = fig.add_subplot(111, projection='3d')

# Plot each cluster separately ONLY for shape, color stays unified
for i in range(best_k):
    cluster_mask = (kmeans_labels == i)
    ax.scatter(
        pca_coords[cluster_mask, 0],
        pca_coords[cluster_mask, 1],
        pca_coords[cluster_mask, 2],
        c=colors[cluster_mask],  # Use shared colors array
        marker=shapes[i % len(shapes)],
        edgecolor='k',
        linewidths=0.4,           # <-- THINNER EDGE
        label=f"Cluster {i}",
        alpha=0.95,                # optional: slightly less transparent
        s=100  # <-- This controls point size (try 50, 70, 100, etc.)
    )

# Colorbar (global score)
mappable = cm.ScalarMappable(norm=norm, cmap=cmap)
mappable.set_array(scores)
cbar = fig.colorbar(mappable, ax=ax, shrink=0.7, pad=0.1)
cbar.set_label('Global Score', fontsize=14)
cbar.ax.tick_params(labelsize=12)

# Axis labels
ax.set_xlabel(f"PC1 ({pca3d_model.explained_variance_ratio_[0]*100:.1f}%)", fontsize=14)
ax.set_ylabel(f"PC2 ({pca3d_model.explained_variance_ratio_[1]*100:.1f}%)", fontsize=14)
ax.set_zlabel(f"PC3 ({pca3d_model.explained_variance_ratio_[2]*100:.1f}%)", fontsize=14)
# Tick label font sizes
ax.tick_params(axis='both', which='major', labelsize=12)

# Title
ax.set_title("3D PCA Clustering — Shape = Cluster, Color = Global Score", fontsize=16)
# # Add variable axes (thin lines with labels)
# loadings = pca3d_model.components_.T * np.sqrt(pca3d_model.explained_variance_)
# for i, var in enumerate(all_criteria):
#     ax.plot([0, loadings[i, 0]], [0, loadings[i, 1]], [0, loadings[i, 2]],
#             color='gray', alpha=0.6, linewidth=0.5)
#     ax.text(loadings[i, 0]*1.3, loadings[i, 1]*1.3, loadings[i, 2]*1.3,
#             var, fontsize=7, ha='center', va='center')

# Legend
ax.legend(fontsize=12)
plt.tight_layout()
plt.savefig(f"{output_dir}/pca3d_clustering_on_pca.png")
plt.show()
plt.close()

# PCA 3D correlation circle
loadings = pca3d_model.components_.T * np.sqrt(pca3d_model.explained_variance_)

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
for i, var in enumerate(all_criteria):
    ax.plot([0, loadings[i, 0]], [0, loadings[i, 1]], [0, loadings[i, 2]],
            color='gray', alpha=0.6, linewidth=0.8)
    ax.text(loadings[i, 0]*1.3, loadings[i, 1]*1.3, loadings[i, 2]*1.3,
            var, color='black', fontsize=8, ha='center', va='center')
ax.set_xlim([-1, 1])
ax.set_ylim([-1, 1])
ax.set_zlim([-1, 1])
ax.set_xlabel(f"PC1 ({pca3d_model.explained_variance_ratio_[0]*100:.1f}%)")
ax.set_ylabel(f"PC2 ({pca3d_model.explained_variance_ratio_[1]*100:.1f}%)")
ax.set_zlabel(f"PC3 ({pca3d_model.explained_variance_ratio_[2]*100:.1f}%)")
ax.set_title("3D PCA Correlation Circle")
plt.savefig(f"{output_dir}/pca3d_correlation_circle.png")
plt.show()
plt.close()

# Save PCA loadings
loadings_df = pd.DataFrame(loadings, index=all_criteria, columns=["PC1", "PC2", "PC3"])
loadings_df.to_csv(f"{output_dir}/pca3d_loadings.csv")


# -------- STEP 7: Outlier Detection (Z-scores) and PCA as Alternative ----------
df['Avg_Likert'] = df[likert_cols].mean(axis=1)
df['Z_100'] = zscore(df[hundred_point_col])
df['Z_Likert'] = zscore(df['Avg_Likert'])

df['Outlier_100'] = df['Z_100'].abs() > 2
df['Outlier_Likert'] = df['Z_Likert'].abs() > 2

outliers = df[df['Outlier_100'] | df['Outlier_Likert']]
outliers[[id_col, hundred_point_col, 'Avg_Likert', 'Z_100', 'Z_Likert', 'Outlier_100', 'Outlier_Likert']] \
    .to_csv(f"{output_dir}/outlier_samples.csv", index=False)

# Alternative: Use PCA distance from center
pca_all = PCA(n_components=3).fit_transform(scaled_features)
distances = np.linalg.norm(pca_all, axis=1)
z_distances = zscore(distances)
df['PCA_Outlier'] = np.abs(z_distances) > 2

df[[id_col, 'PCA_Outlier']].to_csv(f"{output_dir}/pca_outlier_flags.csv", index=False)


from scipy.stats import spearmanr, mannwhitneyu
from statsmodels.stats.multitest import multipletests

# Clone original df to avoid mutation
df_spearman = df.copy()

# Custom recoding for certain MC criteria
custom_map = {
    "Concision": {"A": 0, "B": 1, "C": 0},
    "Redondance": {"A": 0, "B": 1, "C": 0},
    "Voix Passive": {"A": 0, "B": 1, "C": 0},
    "Langage Négatif": {"A": 0, "B": 1, "C": 0},
}
for col, mapping in custom_map.items():
    df_spearman[col] = df_spearman[col].map(mapping)

# Encode all other MC columns (A=0, B=1, C=2)
for col in mc_cols:
    if col not in custom_map:
        df_spearman[col] = df_spearman[col].map({"A": 0, "B": 1, "C": 2})

# --- Spearman Correlation Test ---
spearman_corrs = []

for mc in mc_cols:
    for likert in likert_cols:
        coef, pval = spearmanr(df_spearman[mc], df_spearman[likert])
        spearman_corrs.append({
            "MC_Criterion": mc,
            "Likert_Criterion": likert,
            "Spearman_Coefficient": coef,
            "p_value": pval
        })

spearman_df = pd.DataFrame(spearman_corrs)
# Optional: Adjust for multiple comparisons if needed
spearman_df["p_adj"] = multipletests(spearman_df["p_value"], method="fdr_bh")[1]

spearman_df.to_csv(f"{output_dir}/spearman_mc_vs_likert.csv", index=False)



from scipy.stats import rankdata, norm
import itertools

def jonckheere_terpstra_test(x, y):
    """
    x: group labels (ordered categorical)
    y: continuous values
    Returns: JT statistic, z-score, p-value
    """
    # Ensure arrays
    x = np.array(x)
    y = np.array(y)

    # Get ordered unique groups
    groups = np.unique(x)
    k = len(groups)

    # Build groupwise lists
    data_by_group = [y[x == g] for g in groups]

    # JT statistic
    JT = 0
    for i in range(k):
        for j in range(i + 1, k):
            for a in data_by_group[i]:
                for b in data_by_group[j]:
                    JT += 1 if a < b else 0.5 if a == b else 0

    # Compute variance
    n = np.array([len(g) for g in data_by_group])
    N = np.sum(n)
    mean_JT = np.sum([n[i]*n[j] / 2 for i in range(k) for j in range(i + 1, k)])

    var_JT = (
        np.sum([n[i]*n[j]*(n[i]+n[j]+1)/12 for i in range(k) for j in range(i + 1, k)])
    )

    z = (JT - mean_JT) / np.sqrt(var_JT)
    p_value = 2 * (1 - norm.cdf(abs(z)))

    return JT, z, p_value


jt_results = []

for mc in mc_cols:
    for likert in likert_cols:
        try:
            data = df[[mc, likert]].dropna()
            # Recode MC levels to ordinal integers (e.g., A=0, B=1, C=2)
            mc_numeric = LabelEncoder().fit_transform(data[mc])
            jt_stat, jt_z, jt_p = jonckheere_terpstra_test(mc_numeric, data[likert].values)
            jt_results.append({
                'MC_Criterion': mc,
                'Likert_Criterion': likert,
                'JT_Statistic': jt_stat,
                'Z_score': jt_z,
                'p_value': jt_p
            })
        except Exception as e:
            jt_results.append({
                'MC_Criterion': mc,
                'Likert_Criterion': likert,
                'JT_Statistic': None,
                'Z_score': None,
                'p_value': None,
                'Error': str(e)
            })

jt_df = pd.DataFrame(jt_results)
jt_df.to_csv(f"{output_dir}/jonckheere_trend_test.csv", index=False)
