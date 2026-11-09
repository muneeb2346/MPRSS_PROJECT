"""
train_lstm.py - LSTM vs the simple baseline, judged by the SAME strict tests.

Usage (from the MPRSS_Module1_Behavior folder; needs PyTorch):
    python train_lstm.py                   # real call order
    python train_lstm.py --shuffle-order   # same data, but the order of calls inside each app is scrambled

Why two runs: if scrambling the order does not hurt the LSTM, then ORDER carries no
information in this data and the LSTM is only counting calls like the baseline does.
That is a legitimate finding, and it is better to know it than to claim otherwise.

Tests (identical to baseline_check.py, so the numbers are directly comparable):
  * 5-fold GROUP split: near-duplicate apps never sit on both sides.
  * CROSS-YEAR: train on one year, test on the other.
The LSTM uses part of each TRAINING set (split by groups) to decide when to stop;
the test apps are never used for any decision.
"""

import time
import copy
import argparse
import json

import numpy as np

try:
    import torch
    import torch.nn as nn
    from torch.nn.utils.rnn import pack_padded_sequence
except ImportError:
    raise SystemExit(
        "PyTorch is not installed.\n"
        "Try:  pip install torch --index-url https://download.pytorch.org/whl/cpu\n"
        "If pip says 'No matching distribution found', your Python version has no PyTorch build yet - "
        "tell me, and we will use a separate Python 3.12 environment."
    )

from sklearn.metrics import balanced_accuracy_score
from sklearn.model_selection import StratifiedGroupKFold, GroupShuffleSplit

from baseline_check import bag_of_calls, fit_predict


class SeqClassifier(nn.Module):
    """Embedding -> LSTM -> last hidden state -> 1 score (benign vs malware)."""

    def __init__(self, vocab_size, emb_dim=16, hidden=32, dropout=0.3):
        super().__init__()
        self.emb = nn.Embedding(vocab_size + 1, emb_dim, padding_idx=0)   # id 0 = padding
        self.lstm = nn.LSTM(emb_dim, hidden, batch_first=True)
        self.drop = nn.Dropout(dropout)
        self.out = nn.Linear(hidden, 1)

    def forward(self, x, lengths):
        packed = pack_padded_sequence(self.emb(x), lengths.cpu(), batch_first=True, enforce_sorted=False)
        _, (h, _) = self.lstm(packed)          # h[-1] = state after the LAST real call (padding is ignored)
        return self.out(self.drop(h[-1])).squeeze(1)


def shuffle_within_apps(X, lengths, seed):
    """Scramble the order of the real calls inside each app (padding stays at the end)."""
    rng = np.random.default_rng(seed)
    out = X.copy()
    for i, n in enumerate(lengths):
        out[i, :n] = rng.permutation(X[i, :n])
    return out


@torch.no_grad()
def predict(model, X, L, device):
    model.eval()
    logits = model(torch.as_tensor(X, dtype=torch.long).to(device), torch.as_tensor(L, dtype=torch.long))
    return (torch.sigmoid(logits) > 0.5).long().cpu().numpy()


def train_model(X, L, y, groups, vocab_size, args, device, seed):
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)

    # hold out ~15% of the TRAINING groups to decide when to stop (never the test apps)
    inner_tr, inner_va = next(GroupShuffleSplit(n_splits=1, test_size=0.15, random_state=seed).split(X, y, groups))

    model = SeqClassifier(vocab_size, args.emb, args.hidden, args.dropout).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=1e-4)
    loss_fn = nn.BCEWithLogitsLoss()
    Xt = torch.as_tensor(X, dtype=torch.long)
    Lt = torch.as_tensor(L, dtype=torch.long)
    yt = torch.as_tensor(y, dtype=torch.float32)

    best_state, best_val, bad = None, -1.0, 0
    for epoch in range(args.epochs):
        model.train()
        order = inner_tr[rng.permutation(len(inner_tr))]
        for i in range(0, len(order), args.batch):
            idx = torch.as_tensor(order[i:i + args.batch])
            opt.zero_grad()
            loss = loss_fn(model(Xt[idx].to(device), Lt[idx]), yt[idx].to(device))
            loss.backward()
            opt.step()
        val = balanced_accuracy_score(y[inner_va], predict(model, X[inner_va], L[inner_va], device))
        if val > best_val + 1e-4:
            best_val, bad, best_state = val, 0, copy.deepcopy(model.state_dict())
        else:
            bad += 1
            if bad >= args.patience:
                break
    model.load_state_dict(best_state)
    return model


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="logs/dataset_v1.npz")
    ap.add_argument("--folds", type=int, default=5)
    ap.add_argument("--epochs", type=int, default=60)
    ap.add_argument("--patience", type=int, default=8)
    ap.add_argument("--batch", type=int, default=32)
    ap.add_argument("--lr", type=float, default=3e-3)
    ap.add_argument("--emb", type=int, default=16)
    ap.add_argument("--hidden", type=int, default=32)
    ap.add_argument("--dropout", type=float, default=0.3)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--shuffle-order", action="store_true")
    args = ap.parse_args()

    d = np.load(args.data)
    X, lengths, y, year, group = d["X"], d["lengths"], d["y"], d["year"], d["group"]
    with open(args.data.replace(".npz", "_vocab.json"), encoding="utf-8") as f:
        V = len(json.load(f))
    if args.shuffle_order:
        X = shuffle_within_apps(X, lengths, args.seed)
    B = bag_of_calls(X, V)           # baseline input: how often each call appears (order is invisible to it)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    mode = "SHUFFLED call order" if args.shuffle_order else "real call order"
    print(f"{len(y)} apps, {V} vocabulary APIs, {len(np.unique(group))} groups   [{mode}]   device={device}\n")
    t0 = time.time()

    # ---- 1. group-aware cross-validation ----
    cv = StratifiedGroupKFold(n_splits=args.folds, shuffle=True, random_state=args.seed)
    base_scores, lstm_scores = [], []
    print("1) GROUP-AWARE cross-validation (balanced accuracy, 50% = chance)")
    for k, (tr, te) in enumerate(cv.split(B, y, group), start=1):
        b_pred, _ = fit_predict(B[tr], y[tr], B[te])
        model = train_model(X[tr], lengths[tr], y[tr], group[tr], V, args, device, seed=args.seed + k)
        l_pred = predict(model, X[te], lengths[te], device)
        b, l = balanced_accuracy_score(y[te], b_pred), balanced_accuracy_score(y[te], l_pred)
        base_scores.append(b)
        lstm_scores.append(l)
        print(f"   fold {k}: baseline {b * 100:5.1f}%   LSTM {l * 100:5.1f}%")
    print(f"   MEAN   : baseline {np.mean(base_scores) * 100:5.1f}% (+/- {np.std(base_scores) * 100:.1f})   "
          f"LSTM {np.mean(lstm_scores) * 100:5.1f}% (+/- {np.std(lstm_scores) * 100:.1f})")

    # ---- 2. cross-year ----
    print("\n2) CROSS-YEAR (train on one year, test on the other)")
    for tr_year, te_year in ((2018, 2019), (2019, 2018)):
        tr, te = year == tr_year, year == te_year
        b_pred, _ = fit_predict(B[tr], y[tr], B[te])
        model = train_model(X[tr], lengths[tr], y[tr], group[tr], V, args, device, seed=args.seed + 100)
        l_pred = predict(model, X[te], lengths[te], device)
        print(f"   train {tr_year} -> test {te_year}: baseline {balanced_accuracy_score(y[te], b_pred) * 100:5.1f}%   "
              f"LSTM {balanced_accuracy_score(y[te], l_pred) * 100:5.1f}%")

    print(f"\nDone in {time.time() - t0:.0f}s.")
    print("Please send me everything printed above (as text).")


if __name__ == "__main__":
    main()