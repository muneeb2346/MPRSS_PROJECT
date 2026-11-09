"""
test_behavior_service.py - check that the running Module 1 service is wired correctly.

Usage (service must be running in another window):   python test_behavior_service.py

It sends call sequences taken from the dataset to the service, shaped exactly like the
events Frida will send, and prints the scores next to the true labels.

IMPORTANT: these apps were part of the training data, so the scores are optimistic.
This test proves the PLUMBING works (events in, score out). It is NOT evidence of accuracy;
the honest accuracy numbers are the cross-validation ones in models\\behavior_meta.json.
"""

import json
import argparse

import numpy as np
import requests


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="logs/dataset_v1.npz")
    ap.add_argument("--url", default="http://localhost:5003")
    ap.add_argument("--per-class", type=int, default=5)
    args = ap.parse_args()

    d = np.load(args.data)
    X, lengths, y = d["X"], d["lengths"], d["y"]
    with open(args.data.replace(".npz", "_vocab.json"), encoding="utf-8") as f:
        vocab = json.load(f)
    rng = np.random.default_rng(1)

    health = requests.get(f"{args.url}/api/health", timeout=10).json()
    print(f"Service OK: {health['vocab_size']} monitored APIs, needs >= {health['min_tokens_needed']} calls\n")

    def events_for(i):
        return [{"ts": 1_700_000_000_000 + k, "api": vocab[t - 1]} for k, t in enumerate(X[i, :lengths[i]])]

    print(f"{'true label':<12}{'score':>8}   calls used")
    for label, name in ((0, "benign"), (1, "malware")):
        for i in rng.choice(np.where(y == label)[0], size=args.per_class, replace=False):
            r = requests.post(f"{args.url}/api/analyze/behavior", json={"events": events_for(i)}, timeout=30).json()
            print(f"{name:<12}{r['behavioral_score']!s:>8}   {r['n_tokens_used']}")

    print("\nEdge cases:")
    r = requests.post(f"{args.url}/api/analyze/behavior", json={"events": events_for(0)[:3]}, timeout=30).json()
    print(f"   only 3 calls      -> score={r['behavioral_score']}  ({r['note']})")
    r = requests.post(f"{args.url}/api/analyze/behavior",
                      json={"events": [{"ts": 1, "api": "not.in.Vocabulary.method"}] * 20}, timeout=30).json()
    print(f"   20 unknown calls  -> score={r['behavioral_score']}  (ignored: {r['n_events_received'] - r['n_in_vocabulary']})")
    r = requests.post(f"{args.url}/api/analyze/behavior", json={"wrong": 1}, timeout=30)
    print(f"   bad request       -> HTTP {r.status_code}")


if __name__ == "__main__":
    main()