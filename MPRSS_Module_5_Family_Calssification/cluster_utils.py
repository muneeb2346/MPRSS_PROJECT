"""
MPRSS Module 5 - Shared clustering utilities
Used by both static_family_cluster_trainer.py and flow_family_cluster_trainer.py
so that k-selection (elbow + silhouette), cluster interpretation, and dataset
balancing logic stay IDENTICAL across the two clustering pipelines rather than
being duplicated and potentially drifting apart (see Module 3's log-transform
mismatch between training and inference for what happens when two pipelines
that are supposed to agree quietly diverge).
"""

import os
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score


def select_optimal_k(X, k_range=range(2, 11), random_state=42, sample_size_for_silhouette=5000):
    """
    Fit KMeans for each k in k_range and compute:
      - inertia (within-cluster sum of squares) -> for the elbow plot
      - silhouette score -> for picking a defensible k WITHOUT ground-truth
        family labels (we only have benign/malicious 0/1, not family names)

    Silhouette score is computed on a random subsample (default 5000 rows)
    when X is large, since silhouette_score is O(n^2) and becomes very slow
    on large datasets (the flow dataset's malicious rows can number in the
    hundreds of thousands). The KMeans FIT always uses the full dataset -
    only the silhouette EVALUATION is subsampled.

    Returns
    -------
    results : list of dict, one per k, with keys 'k', 'inertia', 'silhouette'
    best_k  : int, the k with the highest silhouette score
    """
    results = []
    rng = np.random.RandomState(random_state)

    for k in k_range:
        km = KMeans(n_clusters=k, random_state=random_state, n_init=10)
        labels = km.fit_predict(X)

        if len(X) > sample_size_for_silhouette:
            idx = rng.choice(len(X), size=sample_size_for_silhouette, replace=False)
            sil = silhouette_score(X[idx], labels[idx])
        else:
            sil = silhouette_score(X, labels)

        results.append({'k': k, 'inertia': float(km.inertia_), 'silhouette': float(sil)})
        print(f"  k={k:2d}  inertia={km.inertia_:>14.2f}  silhouette={sil:.4f}")

    best = max(results, key=lambda r: r['silhouette'])
    return results, best['k']


def get_cluster_top_features(X_original_df, cluster_labels, feature_names, top_n=10):
    """
    For each cluster, rank features by how much their mean value in that
    cluster deviates from the mean across ALL clustered (malicious) samples.
    Used to write a human-readable INTERPRETATION of each cluster (e.g.
    "Cluster 2: elevated SEND_SMS, READ_CONTACTS -> consistent with
    SMS-trojan/spyware behavior") - this is an interpretation based on
    feature prominence, NOT a verified ground-truth family label. No such
    labels exist in this dataset (only benign/malicious 0/1) - documented
    the same way as Module 3's honest limitations section.

    RANKING METHOD: features are ranked by STANDARDIZED deviation
    (raw deviation / that feature's overall standard deviation), not raw
    deviation. Ranking by raw deviation would let features with naturally
    large absolute scale (e.g. port numbers, 0-65535) mechanically dominate
    the ranking over proportionally much larger, more meaningful shifts on
    smaller-scale features (e.g. a binary flag moving from 0.10 to 0.97 is
    a near-total behavioral change, but its raw deviation of ~0.87 would
    otherwise lose to a port's raw deviation of ~3000, despite the port
    only shifting by a few percent of its range). Both the raw and
    standardized values are still reported, for transparency.

    Parameters
    ----------
    X_original_df : DataFrame of UNSCALED feature values (for interpretability
        - "this permission is present in 80% of this cluster's apps" reads
        better than a z-score on log-transformed, scaled data)
    cluster_labels : array of cluster assignments, same length as X_original_df
    feature_names : list of column names to consider
    top_n : how many top features to report per cluster
    """
    df = X_original_df.copy()
    df['__cluster__'] = cluster_labels

    overall_mean = df[feature_names].mean()
    overall_std = df[feature_names].std().replace(0, 1e-9)  # avoid div-by-zero on constant columns
    interpretations = {}

    for cluster_id in sorted(df['__cluster__'].unique()):
        cluster_df = df[df['__cluster__'] == cluster_id]
        cluster_mean = cluster_df[feature_names].mean()
        deviation = cluster_mean - overall_mean
        standardized_deviation = deviation / overall_std

        ranked = standardized_deviation.abs().sort_values(ascending=False)

        top_features = [
            {
                'feature': feat,
                'cluster_mean': round(float(cluster_mean[feat]), 4),
                'overall_mean': round(float(overall_mean[feat]), 4),
                'deviation': round(float(deviation[feat]), 4),
                'standardized_deviation': round(float(standardized_deviation[feat]), 4),
            }
            for feat in ranked.head(top_n).index
        ]

        interpretations[str(cluster_id)] = {
            'cluster_size': int(len(cluster_df)),
            'cluster_pct': round(len(cluster_df) / len(df) * 100, 2),
            'top_distinguishing_features': top_features,
        }

    return interpretations


def plot_k_selection(results, save_path=None, title_prefix=''):
    """Plot the elbow curve (inertia) and silhouette score side-by-side,
    for use as a figure in the project report."""
    import matplotlib.pyplot as plt

    ks = [r['k'] for r in results]
    inertias = [r['inertia'] for r in results]
    sils = [r['silhouette'] for r in results]

    fig, axes = plt.subplots(1, 2, figsize=(12, 4))

    axes[0].plot(ks, inertias, marker='o')
    axes[0].set_xlabel('k (number of clusters)')
    axes[0].set_ylabel('Inertia (within-cluster SSE)')
    axes[0].set_title(f'{title_prefix}Elbow Method')

    axes[1].plot(ks, sils, marker='o', color='orange')
    axes[1].set_xlabel('k (number of clusters)')
    axes[1].set_ylabel('Silhouette Score')
    axes[1].set_title(f'{title_prefix}Silhouette Score')

    plt.tight_layout()

    if save_path:
        out_dir = os.path.dirname(save_path)
        if out_dir:
            os.makedirs(out_dir, exist_ok=True)
        plt.savefig(save_path, dpi=150)
        print(f"✅ k-selection plot saved to: {os.path.abspath(save_path)}")
    else:
        plt.show()

    plt.close(fig)


def stratified_sample_by_cluster(df, cluster_labels, n_per_cluster=None, random_state=42):
    """
    Sample rows evenly ACROSS clusters, so a downstream training set for
    Module 2/6 isn't dominated by whichever malware pattern happens to be
    most common in the raw dataset. This is Module 5's proposal Function 2:
    "dataset balancing for model training - ensures ML models are trained
    on balanced and diverse malware types."

    n_per_cluster : if None, uses the size of the SMALLEST cluster, so every
        cluster contributes equally (the strictest, most balanced option).
    """
    df = df.copy()
    df['__cluster__'] = cluster_labels
    cluster_sizes = df['__cluster__'].value_counts()

    if n_per_cluster is None:
        n_per_cluster = int(cluster_sizes.min())

    sampled = []
    for cluster_id in sorted(df['__cluster__'].unique()):
        cluster_df = df[df['__cluster__'] == cluster_id]
        n = min(n_per_cluster, len(cluster_df))
        sampled.append(cluster_df.sample(n=n, random_state=random_state))

    balanced_df = pd.concat(sampled, ignore_index=True).drop(columns='__cluster__')
    print(f"Balanced sample: up to {n_per_cluster} rows x {len(cluster_sizes)} clusters "
          f"= {len(balanced_df)} total rows (from {len(df)} original)")
    return balanced_df