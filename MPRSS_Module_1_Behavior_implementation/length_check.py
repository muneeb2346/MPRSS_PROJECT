"""
length_check.py - is trace LENGTH leaking the answer?

Usage (from the MPRSS_Module1_Behavior folder, after explore_apis.py has run):
    python length_check.py

Why: in the first profile, benign apps had a median of 0 framework calls per
trace while malware had a median of ~4,000. If a model can separate the classes
just by how long the trace is, it has learned "did the app run at all", not
"how does the app behave". This script measures that, and shows how many apps
remain if we require a minimum trace length and cut every app to the same
length (so length can no longer give the answer away).
"""

import os
import argparse

import numpy as np


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cache", default=os.path.join("logs", "api_cache.npz"))
    args = ap.parse_args()

    z = np.load(args.cache)
    offsets, labels = z["offsets"], z["labels"]
    lengths = offsets[:, 1] - offsets[:, 0]          # framework calls per app
    ben, mal = lengths[labels == 0], lengths[labels == 1]

    print(f"Apps in cache: benign={len(ben)}  malware={len(mal)}")
    print(f"Apps with ZERO framework calls: benign={np.mean(ben == 0) * 100:.1f}%  "
          f"malware={np.mean(mal == 0) * 100:.1f}%")

    # ---- 1. how well does length ALONE separate the classes? ----
    best_bal, best_t = 0.0, 0
    for t in np.unique(lengths):
        bal = ((mal >= t).mean() + (ben < t).mean()) / 2      # balanced accuracy
        if bal > best_bal:
            best_bal, best_t = bal, int(t)
    print("\nLENGTH-ONLY baseline (predict 'malware' if the trace has >= T framework calls):")
    print(f"   best T = {best_t}  ->  balanced accuracy = {best_bal * 100:.1f}%")
    print("   (50% = no information. Anything far above that means length is leaking the label.)")

    # ---- 2. how many apps survive a minimum-length rule? ----
    print("\nApps with at least K framework calls (every app would then be cut to its first K):")
    print(f"   {'K':>6}{'benign':>9}{'malware':>9}{'balanced set size':>20}")
    for K in (50, 100, 200, 300, 500, 1000, 2000, 5000):
        b, m = int((ben >= K).sum()), int((mal >= K).sum())
        print(f"   {K:>6}{b:>9}{m:>9}{2 * min(b, m):>20}")
    print("\n'balanced set size' = apps you would train on if you keep the same number from each class.")
    print("Please send me everything printed above (as text).")


if __name__ == "__main__":
    main()