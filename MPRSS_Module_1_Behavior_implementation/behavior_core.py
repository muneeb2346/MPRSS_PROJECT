"""
behavior_core.py - Module 1 live scoring logic.

events_to_sequence() turns the events Frida reports into the SAME kind of input the
model was trained on: vocabulary calls only, in time order, at most max_len of them.
It needs no PyTorch, so it can be tested on its own.

BehaviorScorer loads the trained LSTM and returns a 0-100 behavioral suspicion score.
Honesty rule: the model was trained only on apps with at least `min_len` vocabulary calls.
With fewer, it has no basis to judge, so we return behavioral_score = None (never a made-up
number). Module 6 already redistributes weight over whichever scores are available.
"""

import os
import json


def events_to_sequence(events, vocab_index, max_len):
    """
    events      : list of dicts from Frida; each needs 'api' (the full API name, exactly as in
                  vocab_v1.txt). 'ts' (milliseconds) is used for ordering when present.
    vocab_index : dict  API name -> id (1-based)
    Returns (ids, names, n_in_vocab): ids/names = first max_len vocabulary calls in time order;
    n_in_vocab = how many vocabulary calls were seen in total.
    """
    def time_of(pair):
        i, ev = pair
        if not isinstance(ev, dict):
            return (0.0, i)                              # malformed event: skipped later, must not crash the sort
        try:
            return (float(ev.get("ts", 0) or 0), i)      # time first, arrival order breaks ties
        except (TypeError, ValueError):
            return (0.0, i)

    ids, names, n_in_vocab = [], [], 0
    for _, ev in sorted(enumerate(events), key=time_of):
        if not isinstance(ev, dict):
            continue
        idx = vocab_index.get(ev.get("api"))
        if idx is None:
            continue
        n_in_vocab += 1
        if len(ids) < max_len:
            ids.append(idx)
            names.append(ev["api"])
    return ids, names, n_in_vocab


class BehaviorScorer:
    def __init__(self, models_dir="models"):
        import torch                                   # imported here so the rest of the file needs no PyTorch
        from behavior_model import SeqClassifier

        with open(os.path.join(models_dir, "behavior_meta.json"), encoding="utf-8") as f:
            self.meta = json.load(f)
        self.vocab = self.meta["vocab"]
        self.vocab_index = {api: i for i, api in enumerate(self.vocab, start=1)}
        self.max_len = self.meta["max_len"]
        self.min_len = self.meta["min_len"]

        self.model = SeqClassifier(len(self.vocab), self.meta["emb"], self.meta["hidden"], self.meta["dropout"])
        path = os.path.join(models_dir, "behavior_lstm.pt")
        try:
            state = torch.load(path, map_location="cpu", weights_only=True)
        except TypeError:                              # older PyTorch without the weights_only option
            state = torch.load(path, map_location="cpu")
        self.model.load_state_dict(state)
        self.model.eval()
        self._torch = torch

    def score(self, events):
        torch = self._torch
        ids, names, n_in_vocab = events_to_sequence(events, self.vocab_index, self.max_len)
        result = {
            "behavioral_score": None,
            "n_events_received": len(events),
            "n_in_vocabulary": n_in_vocab,
            "n_tokens_used": len(ids),
            "tokens_used_preview": names[:12],
            "note": None,
        }
        if len(ids) < self.min_len:
            result["note"] = (f"Only {len(ids)} monitored API call(s) observed; at least {self.min_len} are needed "
                              f"for a behavioral score. Not scored.")
            return result

        x = torch.zeros((1, self.max_len), dtype=torch.long)
        x[0, :len(ids)] = torch.tensor(ids, dtype=torch.long)
        with torch.no_grad():
            p = torch.sigmoid(self.model(x, torch.tensor([len(ids)], dtype=torch.long))).item()
        result["behavioral_score"] = round(100.0 * p, 1)
        result["note"] = ("Relative suspicion score from an LSTM trained on AndroCT traces; weak supporting "
                          "evidence, not a verdict.")
        return result