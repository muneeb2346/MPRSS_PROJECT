"""
baseline_check.py - is there real, learnable signal in this data? (run BEFORE the LSTM)

Usage (from the MPRSS_Module1_Behavior folder, after build_dataset.py):
    python baseline_check.py

Uses a simple model (logistic regression on how often each vocabulary API appears).
It is a yardstick: an LSTM is only worth reporting if it beats it, and any number
here is only believable because of the three strict tests below.

  1. GROUP split     - near-duplicate apps never sit on both sides of the split.
  2. CROSS-YEAR      - train on one year, test on the OTHER year (different malware generation).
  3. LENGTH-ONLY     - a model that sees only the sequence length: the shortcut we must beat.
"""

import json
import argparse

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import balanced_accuracy_score
from sklearn.model_selection import StratifiedGroupKFold


def bag_of_calls(X, vocab_size):
    counts = np.zeros((X.shape[0], vocab_size))
    for t in range(1, vocab_size + 1):
        counts[:, t - 1] = (X == t).sum(axis=1)
    return np.log1p(counts)


def fit_predict(train_X, train_y, test_X):
    clf = LogisticRegression(max_iter=2000, class_weight="balanced", C=1.0)
    clf.fit(train_X, train_y)
    return clf.predict(test_X), clf


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="logs/dataset_v1.npz")
    ap.add_argument("--folds", type=int, default=5)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    d = np.load(args.data)
    X, lengths, y, year, group = d["X"], d["lengths"], d["y"], d["year"], d["group"]
    with open(args.data.replace(".npz", "_vocab.json"), encoding="utf-8") as f:
        vocab = json.load(f)
    B = bag_of_calls(X, len(vocab))
    Len = np.log1p(lengths).reshape(-1, 1)
    print(f"{len(y)} apps ({int((y == 0).sum())} benign / {int((y == 1).sum())} malware), "
          f"{len(vocab)} vocabulary APIs, {len(np.unique(group))} groups\n")

    # ---- 1. group-aware cross-validation ----
    cv = StratifiedGroupKFold(n_splits=args.folds, shuffle=True, random_state=args.seed)
    scores = {"calls model": [], "length only": []}
    for tr, te in cv.split(B, y, group):
        p, _ = fit_predict(B[tr], y[tr], B[te])
        scores["calls model"].append(balanced_accuracy_score(y[te], p))
        p, _ = fit_predict(Len[tr], y[tr], Len[te])
        scores["length only"].append(balanced_accuracy_score(y[te], p))
    print("1) GROUP-AWARE cross-validation (balanced accuracy, 50% = chance):")
    for name, s in scores.items():
        print(f"   {name:<12} {np.mean(s) * 100:5.1f}%   (+/- {np.std(s) * 100:.1f} across folds)")

    # ---- 2. cross-year ----
    print("\n2) CROSS-YEAR (train on one year, test on the other):")
    for tr_year, te_year in ((2018, 2019), (2019, 2018)):
        tr, te = year == tr_year, year == te_year
        p, _ = fit_predict(B[tr], y[tr], B[te])
        p_len, _ = fit_predict(Len[tr], y[tr], Len[te])
        print(f"   train {tr_year} -> test {te_year}:  calls model {balanced_accuracy_score(y[te], p) * 100:5.1f}%   "
              f"length only {balanced_accuracy_score(y[te], p_len) * 100:5.1f}%")

    # ---- 3. what does the model lean on? ----
    _, clf = fit_predict(B, y, B)
    order = np.argsort(clf.coef_[0])
    print("\n3) APIs the model leans on most (all data):")
    print("   toward MALWARE:")
    for i in order[::-1][:6]:
        print(f"      {clf.coef_[0][i]:+.2f}  {vocab[i]}")
    print("   toward BENIGN:")
    for i in order[:6]:
        print(f"      {clf.coef_[0][i]:+.2f}  {vocab[i]}")
    print("\nPlease send me everything printed above (as text).")


if __name__ == "__main__":
    main()