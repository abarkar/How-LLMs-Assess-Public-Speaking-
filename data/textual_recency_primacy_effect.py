# ./data/hypothesis_test_temporality.py

import os
import pandas as pd
import numpy as np
from scipy.stats import spearmanr, norm

# --- Configuration ---
features_root = '/home/alisa/Documents/GitHubProjects/WillLLMsReplaceUs/features'
annotation_file = '/home/alisa/Documents/GitHubProjects/WillLLMsReplaceUs/data/gpt_annotations.csv'
output_file = '/home/alisa/Documents/GitHubProjects/WillLLMsReplaceUs/data/temporal_analysis/temporal_drift_significance.csv'
slices = ['full', 'beginning', 'middle', 'end']
alpha = 0.05


# --- Load and rerank expert criteria ---
df = pd.read_csv(annotation_file)
id_col = df.columns[0]
mc_cols = df.columns[1:10]
likert_cols = df.columns[10:15]
hundred_point_col = df.columns[15]

# Recode MC questions
custom_map = {
    "Concision": {"A": 0, "B": 1, "C": 0},
    "Redondance": {"A": 0, "B": 1, "C": 0},
    "Voix Passive": {"A": 0, "B": 1, "C": 0},
    "Langage Négatif": {"A": 0, "B": 1, "C": 0},
}
for col, mapping in custom_map.items():
    df[col] = df[col].map(mapping)
for col in mc_cols:
    if col not in custom_map:
        df[col] = df[col].map({"A": 0, "B": 1, "C": 2})

df_criteria = df.set_index(id_col)[list(mc_cols) + list(likert_cols) + [hundred_point_col]]

# --- Helper functions ---
def fisher_z(r):
    if abs(r) == 1:
        r = 0.9999 * np.sign(r)
    return 0.5 * np.log((1 + r) / (1 - r))

def compare_correlations(r1, r2, n1, n2):
    z1, z2 = fisher_z(r1), fisher_z(r2)
    se_diff = np.sqrt(1/(n1 - 3) + 1/(n2 - 3))
    z_stat = (z1 - z2) / se_diff
    p_val = 2 * (1 - norm.cdf(abs(z_stat)))
    return z_stat, p_val

# --- Comparison groups ---
comparison_sets = {
    'raw': {
        'pairs': [('full', 'beginning'), ('full', 'middle'), ('full', 'end'),
                  ('beginning', 'middle'), ('beginning', 'end'), ('middle', 'end')],
        'bonferroni': False
    },
    'bonf_full_vs_rest': {
        'pairs': [('full', 'beginning'), ('full', 'middle'), ('full', 'end')],
        'bonferroni': True
    },
    'bonf_parts_only': {
        'pairs': [('beginning', 'middle'), ('beginning', 'end'), ('middle', 'end')],
        'bonferroni': True
    },
    'bonf_all': {
        'pairs': [('full', 'beginning'), ('full', 'middle'), ('full', 'end'),
                  ('beginning', 'middle'), ('beginning', 'end'), ('middle', 'end')],
        'bonferroni': True
    }
}

# --- Main analysis loop ---
results = []

for category in os.listdir(os.path.join(features_root, 'full')):
    if not category.endswith('.csv'):
        continue

    feature_file = category
    feature_name = category.replace('.csv', '')

    feat_per_slice = {}
    for sl in slices:
        path = os.path.join(features_root, sl, feature_file)
        df_feat = pd.read_csv(path).set_index(id_col)
        feat_per_slice[sl] = df_feat

    for feat in df_feat.columns:
        # Gather correlation and sample sizes
        corrs = {}
        sample_sizes = {}
        for sl in slices:
            merged = feat_per_slice[sl].join(df_criteria, how='inner')
            corrs[sl] = {}
            sample_sizes[sl] = {}
            for crit in df_criteria.columns:
                rho, pval = spearmanr(merged[feat], merged[crit])
                corrs[sl][crit] = {'rho': rho, 'pval': pval}
                sample_sizes[sl][crit] = len(merged)

        for crit in df_criteria.columns:
            row = {
                "Lexical_Feature": feat,
                "Criterion": crit,
                "Category": feature_name
            }


            # Track correlation trend string with correlation values and p-values
            full_rho = corrs['full'][crit]['rho']
            full_pval = corrs['full'][crit]['pval']
            trend_parts = []
            for sl in ['beginning', 'middle', 'end']:
                val = corrs[sl][crit]['rho']
                pval = corrs[sl][crit]['pval']
                if pd.isna(full_rho) or pd.isna(val):
                    direction = "?"
                elif val > full_rho:
                    direction = "↑"
                elif val < full_rho:
                    direction = "↓"
                else:
                    direction = "→"
                trend_parts.append(f"{sl}:{val:.2f} (p={pval:.3f}) {direction}")

            row["Correlation_Trend"] = f"full:{full_rho:.2f} (p={full_pval:.3f}) → " + ", ".join(trend_parts)

            # Run all 4 settings
            for setting, cfg in comparison_sets.items():
                pairs = cfg['pairs']
                bonf = cfg['bonferroni']

                # Step 1: Count eligible comparisons
                eligible_pairs = []
                for s1, s2 in pairs:
                    c1, c2 = corrs[s1][crit], corrs[s2][crit]
                    r1, p1 = c1['rho'], c1['pval']
                    r2, p2 = c2['rho'], c2['pval']
                    n1, n2 = sample_sizes[s1][crit], sample_sizes[s2][crit]

                    if p1 < 1 and p2 < 1 and n1 > 3 and n2 > 3 and not pd.isna(r1) and not pd.isna(r2):
                        eligible_pairs.append((s1, s2, r1, r2, n1, n2))

                eligible_count = len(eligible_pairs)
                corrected_alpha = alpha / eligible_count if bonf and eligible_count > 0 else alpha

                # Step 2: Test correlations only for eligible pairs
                sig_pairs = []
                for s1, s2, r1, r2, n1, n2 in eligible_pairs:
                    z, p = compare_correlations(r1, r2, n1, n2)
                    if p < corrected_alpha:
                        sig_pairs.append(f"{s1}_vs_{s2}")

                # Step 3: Record results
                row[f"Significant_Changes_{setting}"] = "; ".join(sig_pairs)
                row[f"Alpha_{setting}"] = round(corrected_alpha, 5)
                row[f"N_Eligible_Comparisons_{setting}"] = eligible_count

            results.append(row)

# --- Save ---
df_out = pd.DataFrame(results)
df_out.to_csv(output_file, index=False)
print(f"✅ Results saved to: {output_file}")
