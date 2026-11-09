"""
train_final.py - train the Module 1 LSTM that the live service will use.

Usage (from the MPRSS_Module1_Behavior folder; needs PyTorch):
    python train_final.py

Uses the exact training routine (train_model in train_lstm.py) that was evaluated with the
strict tests, with the same settings. About 15% of the groups are held back by that routine to
decide when to stop, so the model never over-trains on the rest.

Creates:
    models\\behavior_lstm.pt      the trained weights
    models\\behavior_meta.json    vocabulary, settings, and the honest evaluation numbers
"""

import os
import json
import argparse
from datetime import datetime

import numpy as np
import torch

from behavior_model import SeqClassifier
from train_lstm import train_model


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="logs/dataset_v1.npz")
    ap.add_argument("--models-dir", default="models")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    # same settings as the evaluated runs of train_lstm.py
    hp = argparse.Namespace(epochs=60, patience=8, batch=32, lr=3e-3, emb=16, hidden=32, dropout=0.3)

    d = np.load(args.data)
    X, lengths, y, year, group = d["X"], d["lengths"], d["y"], d["year"], d["group"]
    with open(args.data.replace(".npz", "_vocab.json"), encoding="utf-8") as f:
        vocab = json.load(f)
    V = len(vocab)
    print(f"Training on {len(y)} apps ({int((y == 0).sum())} benign / {int((y == 1).sum())} malware), {V} monitored APIs")

    model = train_model(X, lengths, y, group, V, hp, torch.device("cpu"), seed=args.seed)

    # the saved weights must load into the architecture the live service uses
    SeqClassifier(V, hp.emb, hp.hidden, hp.dropout).load_state_dict(model.state_dict())

    model.eval()
    with torch.no_grad():
        p = torch.sigmoid(model(torch.as_tensor(X, dtype=torch.long), torch.as_tensor(lengths, dtype=torch.long))).numpy()
    print("\nSanity check ON THE TRAINING APPS (optimistic - the real test numbers are in the evaluation below):")
    print(f"   mean score, benign apps:  {100 * p[y == 0].mean():.1f}")
    print(f"   mean score, malware apps: {100 * p[y == 1].mean():.1f}")

    os.makedirs(args.models_dir, exist_ok=True)
    torch.save(model.state_dict(), os.path.join(args.models_dir, "behavior_lstm.pt"))
    meta = {
        "vocab": vocab,
        "max_len": int(X.shape[1]),
        "min_len": 5,
        "emb": hp.emb, "hidden": hp.hidden, "dropout": hp.dropout,
        "trained_at": datetime.now().isoformat(),
        "training_data": {
            "source": "AndroCT emulator traces, 2018 + 2019 (benign/malware balanced within each year)",
            "apps": int(len(y)),
            "rules": "first 500 framework calls per app; monitored APIs only; <=64 calls; >=5 calls; near-duplicate groups kept together in splits",
        },
        "evaluation": {
            "note": "Balanced accuracy (50% = chance), from train_lstm.py with group-aware folds and cross-year tests",
            "group_aware_cv": {"lstm": 83.5, "lstm_std": 8.7, "count_baseline": 88.0, "baseline_std": 8.8},
            "cross_year": {"2018_to_2019": {"lstm": 68.9, "baseline": 70.3},
                           "2019_to_2018": {"lstm": 70.0, "baseline": 83.4}},
            "order_test": {"lstm_real_order": 83.5, "lstm_shuffled_order": 83.0},
            "limitation": "Order of calls added no measurable information; the simple count model scored higher. Treat the score as weak supporting evidence.",
        },
    }
    with open(os.path.join(args.models_dir, "behavior_meta.json"), "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=1)
    print(f"\nSaved {args.models_dir}\\behavior_lstm.pt and {args.models_dir}\\behavior_meta.json")
    print("Next: python behavior_service.py")


if __name__ == "__main__":
    main()