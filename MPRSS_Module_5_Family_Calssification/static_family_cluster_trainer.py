"""
MPRSS Malware Family Clustering - STATIC side (Module 5, Function 1 & 2)
Groups MALICIOUS apps (Drebin-215 static features) into K-Means clusters
based on permission / API-call / intent / command-signature profiles.

WHY MALICIOUS-ONLY: clustering benign apps into "families" isn't meaningful
- "family" is a malware concept. Including benign rows would just add one
large, irrelevant cluster and distort k-selection.

WHY PCA BEFORE K-MEANS: Drebin-215 features are 215 sparse BINARY columns
(0/1 permission/API flags). K-Means uses Euclidean distance, which in a
high-dimensional sparse binary space tends to cluster mostly by "how many
permissions are active" rather than by WHICH specific ones - a well-known
issue with applying K-Means directly to high-dim sparse binary data. PCA
compresses the 215 dims down to the components that explain most of the
variance (default: 95%), making distances more meaningful. PCA components
are not human-readable on their own, so cluster INTERPRETATION is done
afterward using the ORIGINAL, un-reduced binary features (see cluster_utils).

NO GROUND-TRUTH FAMILY LABELS exist for this dataset (only benign/malicious
0/1), so cluster quality is validated via silhouette score (unsupervised)
and each cluster is labeled with an INTERPRETATION based on its most
distinguishing features - not a verified family name. Documented the same
way as Module 3's and Module 6's honest limitations sections.
"""

import os
import json
import joblib
import numpy as np
from datetime import datetime
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA

from static_feature_extractor import DrebinStaticFeatureExtractor
from cluster_utils import select_optimal_k, get_cluster_top_features, plot_k_selection, stratified_sample_by_cluster


class StaticFamilyClusterTrainer:
    """Train and evaluate K-Means clustering over malicious Drebin-215 apps"""

    def __init__(self, random_state=42):
        self.random_state = random_state
        self.pca = None
        self.kmeans = None
        self.feature_columns = []
        self.X_malicious_raw = None
        self.k_selection_results = []
        self.best_k = None
        self.k_used = None
        self.interpretations = {}

    # ------------------------------------------------------------------
    def prepare_data(self, csv_path, categories_path, pca_variance=0.95):
        """Load Drebin-215 via the existing Module 2 extractor, keep malicious
        rows only, and reduce dimensionality with PCA before clustering."""
        extractor = DrebinStaticFeatureExtractor()
        extractor.load_dataset(csv_path)
        extractor.load_feature_categories(categories_path)
        extractor.clean_dataset(drop_exact_duplicates=False)

        X, y = extractor.split_features_labels()
        self.feature_columns = extractor.feature_columns

        malicious_mask = (y == 1)
        self.X_malicious_raw = X[malicious_mask].reset_index(drop=True)

        print(f"\n{'='*50}")
        print(f"Malicious samples for clustering: {len(self.X_malicious_raw)}")
        print(f"{'='*50}")

        self.pca = PCA(n_components=pca_variance, random_state=self.random_state)
        X_pca = self.pca.fit_transform(self.X_malicious_raw.values)

        print(f"PCA: {X.shape[1]} features -> {X_pca.shape[1]} components "
              f"(retaining {pca_variance*100:.0f}% variance)")

        return X_pca

    # ------------------------------------------------------------------
    def select_k(self, X_pca, k_range=range(2, 11)):
        print(f"\n{'='*50}")
        print("Selecting optimal k (elbow + silhouette)")
        print(f"{'='*50}")
        self.k_selection_results, self.best_k = select_optimal_k(
            X_pca, k_range=k_range, random_state=self.random_state
        )
        print(f"\nBest k by silhouette score: {self.best_k}")
        return self.best_k

    # ------------------------------------------------------------------
    def train(self, X_pca, k=None):
        k = k or self.best_k
        if k is None:
            raise ValueError("No k specified and select_k() has not been run.")

        print(f"\n{'='*50}")
        print(f"Training final KMeans with k={k}")
        print(f"{'='*50}")

        self.kmeans = KMeans(n_clusters=k, random_state=self.random_state, n_init=10)
        cluster_labels = self.kmeans.fit_predict(X_pca)

        print(f"Cluster sizes: {np.bincount(cluster_labels).tolist()}")

        self.interpretations = get_cluster_top_features(
            self.X_malicious_raw, cluster_labels, self.feature_columns, top_n=10
        )
        self._cluster_labels = cluster_labels
        self.k_used = k  # the k actually trained with — may differ from self.best_k
                          # (silhouette's argmax) if it was manually overridden, e.g.
                          # when silhouette doesn't converge to a clear peak within
                          # the tested range (see static clustering's k-selection notes)

        return cluster_labels

    # ------------------------------------------------------------------
    def print_interpretations(self):
        print(f"\n{'='*50}")
        print("Cluster interpretations (feature-based, NOT verified family names)")
        print(f"{'='*50}")
        for cluster_id, info in self.interpretations.items():
            print(f"\nCluster {cluster_id} \u2014 {info['cluster_size']} samples "
                  f"({info['cluster_pct']}% of malicious set)")
            for feat in info['top_distinguishing_features'][:5]:
                print(f"    {feat['feature']:<40} cluster={feat['cluster_mean']:.2f}  "
                      f"overall={feat['overall_mean']:.2f}  (+{feat['deviation']:.2f})")

    # ------------------------------------------------------------------
    def get_balanced_training_sample(self, n_per_cluster=None):
        """Module 5 Function 2: dataset balancing. Returns a version of the
        malicious set sampled evenly across the discovered clusters, for use
        when retraining Module 2/6 so no single malware pattern dominates."""
        return stratified_sample_by_cluster(
            self.X_malicious_raw, self._cluster_labels,
            n_per_cluster=n_per_cluster, random_state=self.random_state
        )

    # ------------------------------------------------------------------
    def save_model(self, model_path='models/static_family_kmeans.pkl'):
        out_dir = os.path.dirname(model_path)
        if out_dir:
            os.makedirs(out_dir, exist_ok=True)
        joblib.dump(self.kmeans, model_path)
        print(f"\n✅ KMeans model saved to: {os.path.abspath(model_path)}")

    def save_pca(self, pca_path='models/static_family_pca.pkl'):
        out_dir = os.path.dirname(pca_path)
        if out_dir:
            os.makedirs(out_dir, exist_ok=True)
        joblib.dump(self.pca, pca_path)
        print(f"✅ PCA transformer saved to: {os.path.abspath(pca_path)}")

    def save_metadata(self, path='models/static_family_metadata.json'):
        out_dir = os.path.dirname(path)
        if out_dir:
            os.makedirs(out_dir, exist_ok=True)
        metadata = {
            'feature_columns': self.feature_columns,
            'best_k_by_silhouette_max': self.best_k,
            'k_used_for_final_model': self.k_used,
            'k_selection_results': self.k_selection_results,
            'pca_n_components': int(self.pca.n_components_),
            'pca_explained_variance': round(float(sum(self.pca.explained_variance_ratio_)), 4),
            'cluster_interpretations': self.interpretations,
            'trained_at': datetime.now().isoformat(),
            'note': ('Cluster labels are INTERPRETATIONS based on distinguishing '
                     'static features, not verified ground-truth family names - '
                     'no labeled family dataset was available for this project. '
                     'k_used_for_final_model may differ from best_k_by_silhouette_max: '
                     'silhouette score did not converge to a clear global maximum '
                     'within the tested range (consistent with high-dimensional '
                     'sparse feature spaces), so a strong LOCAL peak (k=7) was chosen '
                     'manually instead of the global argmax, for interpretability.')
        }
        with open(path, 'w') as f:
            json.dump(metadata, f, indent=2)
        print(f"✅ Metadata saved to: {os.path.abspath(path)}")


# ----------------------------------------------------------------------
if __name__ == "__main__":
    trainer = StaticFamilyClusterTrainer(random_state=42)

    X_pca = trainer.prepare_data(
        csv_path='datasets/drebin215/drebin-215-dataset-5560malware-9476-benign.csv',
        categories_path='datasets/drebin215/dataset-features-categories.csv',
        pca_variance=0.95
    )

    best_k = trainer.select_k(X_pca, k_range=range(2, 21))
    plot_k_selection(trainer.k_selection_results,
                      save_path='logs/static_family_k_selection.png',
                      title_prefix='Static Family Clustering - ')

    trainer.train(X_pca, k=7)
    trainer.print_interpretations()

    trainer.save_model('models/static_family_kmeans.pkl')
    trainer.save_pca('models/static_family_pca.pkl')
    trainer.save_metadata('models/static_family_metadata.json')

    print(f"\n{'='*50}")
    print("✅ Module 5 (static clustering) training complete!")
    print(f"{'='*50}")