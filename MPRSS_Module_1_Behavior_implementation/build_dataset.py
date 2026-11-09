"""
build_dataset.py - turn the cached AndroCT traces into Module 1's training set.

Usage (from the MPRSS_Module1_Behavior folder):
    python build_dataset.py

Every rule below closes a loophole found while auditing the data:
  1. Only apps with at least N framework calls, and only each app's FIRST N calls,
     so every app is observed equally long (trace length can't give the answer away).
  2. Keep only calls to the APIs in vocab_v1.txt (the APIs Frida will hook live), in order.
  3. Benign and malware are balanced WITHIN each year (the year can't give the answer away).
  4. Each app keeps at most L vocabulary calls (same cap will apply to live runs).
  5. Apps are clustered into near-duplicate GROUPS. Splits keep a whole group on one side,
     so near-identical malware can't appear in both the training and the test set.

Output (default logs\\):
    dataset_v1.npz         X (apps x L, 0 = padding), lengths, y (0 benign / 1 malware), year, group
    dataset_v1_vocab.json  token id (1..) -> API name
"""

import os
import json
import argparse

import numpy as np
from sklearn.cluster import MiniBatchKMeans


def read_vocab(path):
    names = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            s = line.strip()
            if s and not s.startswith("#") and s not in names:
                names.append(s)
    return names


def load_cache(cache_dir):
    z = np.load(os.path.join(cache_dir, "api_cache.npz"))
    with open(os.path.join(cache_dir, "api_vocab.json"), encoding="utf-8") as f:
        vocab = json.load(f)
    return z["tokens"], z["offsets"], z["labels"], vocab


def collect_year(cache_dir, vocab_names, n, min_len, max_len, gid_of):
    """Return one record per eligible app: label, its vocabulary sequence, and its API set (for grouping)."""
    tokens, offsets, labels, cvocab = load_cache(cache_dir)
    index = {a: i for i, a in enumerate(cvocab)}

    lut = np.zeros(len(cvocab), dtype=np.int16)            # cache id -> vocabulary id (0 = not in vocabulary)
    missing = []
    for j, name in enumerate(vocab_names, start=1):
        if name in index:
            lut[index[name]] = j
        else:
            missing.append(name)

    cache_to_gid = np.empty(len(cvocab), dtype=np.int32)    # cache id -> id shared by both years (for grouping)
    for i, api in enumerate(cvocab):
        cache_to_gid[i] = gid_of.setdefault(api, len(gid_of))

    lengths = offsets[:, 1] - offsets[:, 0]
    apps = []
    for i in np.where(lengths >= n)[0]:
        start = offsets[i, 0]
        window = tokens[start:start + n]
        mapped = lut[window]
        vseq = mapped[mapped > 0]
        if len(vseq) < min_len:
            continue
        apps.append({"label": int(labels[i]), "vseq": vseq[:max_len], "gids": np.unique(cache_to_gid[window])})
    return apps, missing


def best_balanced_accuracy(lengths, y):
    best = 0.0
    for t in np.unique(lengths):
        bal = ((lengths[y == 1] >= t).mean() + (lengths[y == 0] < t).mean()) / 2
        best = max(best, bal)
    return best


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cache2018", default="logs2018")
    ap.add_argument("--cache2019", default="logs")
    ap.add_argument("--vocab", default="vocab_v1.txt")
    ap.add_argument("--n", type=int, default=500, help="observe the first N framework calls of every app")
    ap.add_argument("--max-len", type=int, default=64, help="keep at most this many vocabulary calls per app")
    ap.add_argument("--min-len", type=int, default=5, help="drop apps with fewer vocabulary calls than this")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default=os.path.join("logs", "dataset_v1.npz"))
    args = ap.parse_args()
    rng = np.random.default_rng(args.seed)

    vocab_names = read_vocab(args.vocab)
    print(f"Vocabulary: {len(vocab_names)} APIs from {args.vocab}")

    gid_of = {}
    years = {}
    for year, d in ((2018, args.cache2018), (2019, args.cache2019)):
        apps, missing = collect_year(d, vocab_names, args.n, args.min_len, args.max_len, gid_of)
        years[year] = apps
        if missing:
            print(f"   {year}: {len(missing)} vocabulary APIs never appear in that cache (ignored): {missing[:3]}...")

    # ---- balance classes within each year ----
    chosen = []
    print(f"\nEligible apps (>= {args.n} framework calls and >= {args.min_len} vocabulary calls) -> balanced:")
    for year, apps in years.items():
        ben = [a for a in apps if a["label"] == 0]
        mal = [a for a in apps if a["label"] == 1]
        k = min(len(ben), len(mal))
        print(f"   {year}: benign={len(ben)}  malware={len(mal)}   -> keep {k} of each")
        if k == 0:
            raise SystemExit(f"No usable apps for one class in {year}. Try a smaller --n or --min-len.")
        for pool in (ben, mal):
            for idx in rng.choice(len(pool), size=k, replace=False):
                chosen.append((year, pool[idx]))

    n_apps, L = len(chosen), args.max_len
    X = np.zeros((n_apps, L), dtype=np.int16)
    lengths = np.zeros(n_apps, dtype=np.int32)
    y = np.zeros(n_apps, dtype=np.int8)
    year_arr = np.zeros(n_apps, dtype=np.int16)
    for i, (year, a) in enumerate(chosen):
        X[i, :len(a["vseq"])] = a["vseq"]
        lengths[i] = len(a["vseq"])
        y[i] = a["label"]
        year_arr[i] = year

    # ---- near-duplicate groups: cluster apps by which of the ~300 most common APIs they call ----
    df = np.zeros(len(gid_of), dtype=np.int64)
    for _, a in chosen:
        df[a["gids"]] += 1
    top = np.argsort(-df)[:300]
    col_of = np.full(len(gid_of), -1, dtype=np.int32)
    col_of[top] = np.arange(len(top))
    presence = np.zeros((n_apps, len(top)), dtype=np.float32)
    for i, (_, a) in enumerate(chosen):
        cols = col_of[a["gids"]]
        presence[i, cols[cols >= 0]] = 1.0
    n_groups = max(10, n_apps // 30)
    group = MiniBatchKMeans(n_clusters=n_groups, random_state=args.seed, n_init=3, batch_size=1024).fit_predict(presence)

    # ---- report ----
    print(f"\nFinal set: {n_apps} apps = {int((y == 0).sum())} benign + {int((y == 1).sum())} malware "
          f"({', '.join(f'{yr}: {int((year_arr == yr).sum())}' for yr in years)})")
    for name, lab in (("benign", 0), ("malware", 1)):
        v = lengths[y == lab]
        print(f"vocabulary calls per app, {name:<8} p10={np.percentile(v, 10):.0f}  median={np.median(v):.0f}  "
              f"p90={np.percentile(v, 90):.0f}")
    print(f"LENGTH-ONLY baseline on these sequences: balanced accuracy = {best_balanced_accuracy(lengths, y) * 100:.1f}%  "
          f"(50% = no leak)")
    sizes = np.bincount(group)
    print(f"near-duplicate groups: {n_groups}   largest group = {sizes.max()} apps ({sizes.max() / n_apps * 100:.0f}% of the set)")

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    np.savez_compressed(args.out, X=X, lengths=lengths, y=y, year=year_arr, group=group)
    vocab_path = args.out.replace(".npz", "_vocab.json")
    with open(vocab_path, "w", encoding="utf-8") as f:
        json.dump(vocab_names, f, indent=1)
    print(f"\nSaved {args.out} and {vocab_path}")
    print("Please send me everything printed above (as text).")


if __name__ == "__main__":
    main()