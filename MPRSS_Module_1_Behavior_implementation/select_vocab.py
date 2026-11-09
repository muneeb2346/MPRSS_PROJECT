"""
select_vocab.py - choose which Android APIs Module 1 should look at.

Usage (from the MPRSS_Module1_Behavior folder):
    python select_vocab.py

Uses the caches written by explore_apis.py:
    logs\\       (2019 traces)      logs2018\\   (2018 traces)

Principle: the model's vocabulary must be APIs that Frida can hook LIVE, so
training and live scoring see the same tokens. This script ranks candidates; we
then hook the chosen APIs in behavior_hooks.js.

How candidates are ranked (and why):
  * Only apps whose trace has at least N framework calls are used, and only the
    FIRST N calls of each, so every app is equally "exposed" and trace length
    cannot decide the ranking.
  * Rates are computed per YEAR. The two years contain different malware
    (2018 has a large group sharing Runtime.exec; 2019 does not), so an API is
    only marked consistent (OK) if it leans the same way in both years.
  * Generic APIs (strings, collections, UI, graphics) are dropped: they say
    nothing about behavior.
  * "Chatty" APIs (called very many times per app) are dropped: hooking them
    would slow the app down and flood the event stream.
"""

import os
import csv
import json
import argparse

import numpy as np

# APIs behavior_hooks.js records today
FRIDA_HOOKED = {
    "android.app.ContextImpl.checkSelfPermission", "android.content.Context.checkSelfPermission",
    "android.telephony.SmsManager.sendTextMessage", "android.content.ContentResolver.query",
    "android.hardware.Camera.open", "android.hardware.camera2.CameraManager.openCamera",
    "android.location.LocationManager.getLastKnownLocation",
    "android.telephony.TelephonyManager.getDeviceId", "android.telephony.TelephonyManager.getSubscriberId",
    "android.telephony.TelephonyManager.getSimSerialNumber", "android.telephony.TelephonyManager.getLine1Number",
    "android.telephony.TelephonyManager.getSimCountryIso",
    "java.net.URL.openConnection", "java.io.FileOutputStream.<init>", "java.lang.Runtime.exec",
}

# Generic / UI / bookkeeping APIs: they tell us the app ran, not what it does
GENERIC_PREFIXES = (
    "java.lang.String", "java.lang.Object", "java.lang.Integer", "java.lang.Long", "java.lang.Boolean",
    "java.lang.Math", "java.lang.Character", "java.lang.Number", "java.lang.Enum", "java.lang.Float",
    "java.lang.Double", "java.lang.Iterable", "java.lang.Comparable", "java.lang.CharSequence",
    "java.lang.Throwable", "java.lang.Exception", "java.lang.System.arraycopy", "java.lang.System.currentTimeMillis",
    "java.util.", "java.text.", "android.util.", "android.view.", "android.graphics.", "android.widget.",
    "android.animation.", "android.content.res.", "android.text.", "android.os.Bundle", "android.os.BaseBundle",
    "android.os.Handler", "android.os.Looper", "android.os.Message", "android.os.SystemClock",
    "android.app.Activity.", "android.app.Fragment", "android.app.Dialog", "android.app.AlertDialog",
    "android.media.", "android.database.", "android.preference.",
)


def load_cache(cache_dir):
    z = np.load(os.path.join(cache_dir, "api_cache.npz"))
    with open(os.path.join(cache_dir, "api_vocab.json"), encoding="utf-8") as f:
        vocab = json.load(f)
    return z["tokens"], z["offsets"], z["labels"], vocab


def year_stats(cache_dir, n):
    """Per-API: % of benign apps / % of malware apps that call it within their first n calls,
    plus average occurrences per app. Only apps with >= n framework calls are used."""
    tokens, offsets, labels, vocab = load_cache(cache_dir)
    lengths = offsets[:, 1] - offsets[:, 0]
    keep = np.where(lengths >= n)[0]
    v = len(vocab)
    present = {0: np.zeros(v), 1: np.zeros(v)}
    occurrences = np.zeros(v)
    n_apps = {0: 0, 1: 0}
    for i in keep:
        start = offsets[i, 0]
        counts = np.bincount(tokens[start:start + n], minlength=v)
        lab = int(labels[i])
        present[lab] += counts > 0
        occurrences += counts
        n_apps[lab] += 1
    total = max(n_apps[0] + n_apps[1], 1)
    out = {}
    for api_id, api in enumerate(vocab):
        out[api] = (
            100.0 * present[0][api_id] / max(n_apps[0], 1),
            100.0 * present[1][api_id] / max(n_apps[1], 1),
            occurrences[api_id] / total,
        )
    return out, n_apps


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cache2018", default="logs2018")
    ap.add_argument("--cache2019", default="logs")
    ap.add_argument("--n", type=int, default=1000, help="use the first N framework calls of each app")
    ap.add_argument("--min-support", type=float, default=2.0, help="min %% of apps (any class/year) that call the API")
    ap.add_argument("--max-rate", type=float, default=20.0, help="drop APIs called more than this many times per app (within N calls)")
    ap.add_argument("--top", type=int, default=50)
    ap.add_argument("--out", default=os.path.join("logs", "vocab_candidates.csv"))
    args = ap.parse_args()

    s18, n18 = year_stats(args.cache2018, args.n)
    s19, n19 = year_stats(args.cache2019, args.n)
    print(f"Apps with >= {args.n} framework calls (only these are used):")
    print(f"   2018: benign={n18[0]}  malware={n18[1]}")
    print(f"   2019: benign={n19[0]}  malware={n19[1]}")
    if min(n18[0], n18[1], n19[0], n19[1]) == 0:
        print("A year/class has no usable apps at this N - try a smaller --n.")
        return

    rows, flipped, generic, chatty, rare = [], 0, 0, 0, 0
    for api in sorted(set(s18) | set(s19)):
        if api.startswith(GENERIC_PREFIXES):
            generic += 1
            continue
        b18, m18, o18 = s18.get(api, (0.0, 0.0, 0.0))
        b19, m19, o19 = s19.get(api, (0.0, 0.0, 0.0))
        rate = (o18 * (n18[0] + n18[1]) + o19 * (n19[0] + n19[1])) / (n18[0] + n18[1] + n19[0] + n19[1])
        if rate > args.max_rate:
            chatty += 1
            continue
        if max(b18, m18, b19, m19) < args.min_support:
            rare += 1
            continue
        d18, d19 = m18 - b18, m19 - b19
        consistent = (d18 > 0 and d19 > 0) or (d18 < 0 and d19 < 0)
        if not consistent:
            flipped += 1
        rows.append((api, b18, m18, b19, m19, (d18 + d19) / 2, rate, consistent))

    print(f"\nDropped: {generic} generic APIs, {chatty} chatty APIs, {rare} rare APIs (< {args.min_support}% support).")
    print(f"Kept {len(rows)} candidates; {flipped} of them lean opposite ways in the two years (marked --).")

    rows.sort(key=lambda r: abs(r[5]), reverse=True)
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    with open(args.out, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["api", "benign_2018_pct", "malware_2018_pct", "benign_2019_pct", "malware_2019_pct",
                    "mean_malware_minus_benign", "calls_per_app", "same_direction_both_years", "already_hooked"])
        for r in rows:
            w.writerow([r[0], f"{r[1]:.1f}", f"{r[2]:.1f}", f"{r[3]:.1f}", f"{r[4]:.1f}",
                        f"{r[5]:.1f}", f"{r[6]:.2f}", r[7], r[0] in FRIDA_HOOKED])

    print(f"\nTop {args.top} candidates, % of apps that call the API   (* = your Frida hooks already record it)")
    print(f"   {'API':<58}{'ben18':>6}{'mal18':>6}{'ben19':>6}{'mal19':>6}{'calls/app':>10}  dir")
    shown = 0
    for r in rows:
        if shown >= args.top:
            break
        mark = "*" if r[0] in FRIDA_HOOKED else " "
        print(f"{mark}  {r[0][:56]:<56}{r[1]:>6.0f}{r[2]:>6.0f}{r[3]:>6.0f}{r[4]:>6.0f}{r[6]:>10.2f}  {'OK' if r[7] else '--'}")
        shown += 1
    print(f"\nFull list saved to {args.out}")
    print("Please send me everything printed above (as text).")


if __name__ == "__main__":
    main()