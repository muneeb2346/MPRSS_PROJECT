"""
explore_apis.py - profile AndroCT traces to decide Module 1's vocabulary.

Usage (from the MPRSS_Module1_Behavior folder):
    python explore_apis.py datasets\\androct\\trace-benign-2019.tar.gz datasets\\androct\\trace-malware-2019.tar.gz

What a trace line looks like (one call per line, no timestamps):
    <com.some.App: void onCreate()> -> <android.telephony.SmsManager: void sendTextMessage(...)>
         caller (the app's own code)          callee (what it calls)

Design decisions this script supports:
  * App-specific class names (com.some.App...) are unique to each app, so they
    cannot be shared tokens across apps. Only calls INTO the Android/Java
    framework are kept as tokens (e.g. android.telephony.SmsManager.sendTextMessage).
  * android.support.* is NOT the platform - it is library code bundled inside
    each app - so it is excluded.
  * The script shows how often each framework API appears in benign vs malware
    apps, and how many apps use the 13 APIs your Frida hooks currently record.
    The final model vocabulary must be APIs Frida can hook, so training and
    live scoring use the same tokens (same lesson as Module 3's preprocessing).

Outputs (in --out-dir, default logs\\):
    api_profile.csv   per-API app counts for benign vs malware
    api_cache.npz     compact integer token sequences for every app (so later
                      steps do not have to re-read the big archives)
    api_vocab.json    id -> API name
The cache is derived from AndroCT: keep it local, do not redistribute it.
"""

import os
import sys
import csv
import json
import time
import tarfile
import argparse
import collections
from array import array

import numpy as np

FRAMEWORK_PREFIXES = ("android.", "java.", "javax.", "dalvik.",
                      "org.apache.http.", "org.json.", "org.xml.", "org.w3c.", "libcore.")
EXCLUDE_PREFIXES = ("android.support.", "android.arch.", "android.databinding.")

# The APIs behavior_hooks.js records today (plus one naming variant, see note in output)
FRIDA_HOOKED = [
    "android.app.ContextImpl.checkSelfPermission",
    "android.content.Context.checkSelfPermission",      # variant: call sites usually name Context, not ContextImpl
    "android.telephony.SmsManager.sendTextMessage",
    "android.content.ContentResolver.query",
    "android.hardware.Camera.open",
    "android.hardware.camera2.CameraManager.openCamera",
    "android.location.LocationManager.getLastKnownLocation",
    "android.telephony.TelephonyManager.getDeviceId",
    "android.telephony.TelephonyManager.getSubscriberId",
    "android.telephony.TelephonyManager.getSimSerialNumber",
    "android.telephony.TelephonyManager.getLine1Number",
    "android.telephony.TelephonyManager.getSimCountryIso",
    "java.net.URL.openConnection",
    "java.io.FileOutputStream.<init>",
    "java.lang.Runtime.exec",
]


def parse_callee_api(line):
    """'<A: ret m(args)> -> <B: ret n(args)>'  ->  'B.n' for framework callees, else None."""
    i = line.find(" -> <")
    if i == -1:
        return None
    callee = line[i + 5:]
    c = callee.find(": ")
    if c == -1:
        return None
    cls = callee[:c]
    if not cls.startswith(FRAMEWORK_PREFIXES) or cls.startswith(EXCLUDE_PREFIXES):
        return None
    rest = callee[c + 2:]
    p = rest.find("(")
    if p == -1:
        return None
    name = rest[:p].rsplit(" ", 1)[-1]
    return f"{cls}.{name}"


def profile_archive(path, label, max_lines, vocab, tokens, offsets, labels):
    """Stream one archive. Fills the shared token cache; returns per-class statistics."""
    stats = {"logcat_files": 0, "no_calls": 0, "calls_per_app": [], "tokens_per_app": []}
    api_df = collections.Counter()     # API -> number of apps that call it
    t0 = time.time()

    with tarfile.open(path, "r:gz") as tar:
        for member in tar:
            if not member.isfile() or not member.name.endswith(".logcat"):
                continue
            stats["logcat_files"] += 1
            n_calls = 0
            apis_in_app = set()
            start = len(tokens)

            with tar.extractfile(member) as f:
                for i, raw in enumerate(f):
                    if i >= max_lines:
                        break
                    line = raw.decode("utf-8", "replace")
                    if line.startswith("---------"):
                        continue
                    n_calls += 1
                    api = parse_callee_api(line)
                    if api:
                        apis_in_app.add(api)
                        tokens.append(vocab.setdefault(api, len(vocab)))

            if n_calls == 0:
                stats["no_calls"] += 1
                continue

            offsets.append((start, len(tokens)))
            labels.append(label)
            stats["calls_per_app"].append(n_calls)
            stats["tokens_per_app"].append(len(tokens) - start)
            api_df.update(apis_in_app)

            if stats["logcat_files"] % 200 == 0:
                print(f"   ...{stats['logcat_files']} traces read ({time.time() - t0:.0f}s)")

    return stats, api_df


def describe(name, stats):
    n = len(stats["calls_per_app"])
    print(f"\n--- {name} ---")
    print(f"trace files (.logcat):        {stats['logcat_files']}")
    print(f"with no call lines (unusable): {stats['no_calls']}")
    print(f"usable apps:                   {n}")
    if n:
        for label, key in (("call lines per app", "calls_per_app"), ("framework tokens per app", "tokens_per_app")):
            v = np.array(stats[key])
            print(f"{label:<26} p10={np.percentile(v, 10):.0f}  median={np.percentile(v, 50):.0f}  "
                  f"p90={np.percentile(v, 90):.0f}  max={v.max()}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("benign_archive")
    ap.add_argument("malware_archive")
    ap.add_argument("--max-lines", type=int, default=200000,
                    help="read at most this many lines of each trace (huge traces reach 1 GB)")
    ap.add_argument("--min-support-pct", type=float, default=3.0,
                    help="an API must appear in at least this %% of apps of one class to be ranked")
    ap.add_argument("--out-dir", default="logs")
    args = ap.parse_args()
    os.makedirs(args.out_dir, exist_ok=True)

    vocab, tokens, offsets, labels = {}, array("i"), [], []

    print("Reading benign traces...")
    ben_stats, ben_df = profile_archive(args.benign_archive, 0, args.max_lines, vocab, tokens, offsets, labels)
    print("Reading malware traces...")
    mal_stats, mal_df = profile_archive(args.malware_archive, 1, args.max_lines, vocab, tokens, offsets, labels)

    describe("BENIGN", ben_stats)
    describe("MALWARE", mal_stats)

    n_ben, n_mal = len(ben_stats["calls_per_app"]), len(mal_stats["calls_per_app"])
    if n_ben == 0 or n_mal == 0:
        print("\nOne class has no usable apps - cannot compare. Send me this output.")
        return

    # ---- per-API comparison ----
    rows = []
    for api in set(ben_df) | set(mal_df):
        b, m = ben_df.get(api, 0), mal_df.get(api, 0)
        bp, mp = 100.0 * b / n_ben, 100.0 * m / n_mal
        rows.append((api, b, m, bp, mp, mp - bp))
    rows.sort(key=lambda r: r[5], reverse=True)

    with open(os.path.join(args.out_dir, "api_profile.csv"), "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["api", "benign_apps", "malware_apps", "benign_pct", "malware_pct", "malware_minus_benign_pct"])
        for r in rows:
            w.writerow([r[0], r[1], r[2], f"{r[3]:.2f}", f"{r[4]:.2f}", f"{r[5]:.2f}"])

    ranked = [r for r in rows if max(r[3], r[4]) >= args.min_support_pct]
    print(f"\nDistinct framework APIs seen: {len(rows)}   (ranked below: {len(ranked)} with >= {args.min_support_pct}% support)")

    print("\nMost MALWARE-leaning APIs (% of apps that call them):")
    print(f"   {'API':<75}{'benign%':>8}{'malware%':>9}")
    for r in ranked[:20]:
        print(f"   {r[0][:74]:<75}{r[3]:>8.1f}{r[4]:>9.1f}")

    print("\nMost BENIGN-leaning APIs:")
    for r in ranked[::-1][:10]:
        print(f"   {r[0][:74]:<75}{r[3]:>8.1f}{r[4]:>9.1f}")

    # ---- coverage of the current Frida hooks ----
    print("\nCoverage of your CURRENT Frida hooks (% of apps whose trace calls this API):")
    print(f"   {'API':<60}{'benign%':>8}{'malware%':>9}")
    by_api = {r[0]: r for r in rows}
    for api in FRIDA_HOOKED:
        r = by_api.get(api)
        bp, mp = (r[3], r[4]) if r else (0.0, 0.0)
        print(f"   {api:<60}{bp:>8.1f}{mp:>9.1f}")

    # ---- save the cache ----
    np.savez_compressed(
        os.path.join(args.out_dir, "api_cache.npz"),
        tokens=np.frombuffer(tokens, dtype=np.intc),
        offsets=np.array(offsets, dtype=np.int64),
        labels=np.array(labels, dtype=np.int8),
    )
    id_to_api = [None] * len(vocab)
    for api, i in vocab.items():
        id_to_api[i] = api
    with open(os.path.join(args.out_dir, "api_vocab.json"), "w", encoding="utf-8") as f:
        json.dump(id_to_api, f)
    print(f"\nSaved to '{args.out_dir}': api_profile.csv, api_cache.npz, api_vocab.json")
    print("Please send me everything printed above (as text).")


if __name__ == "__main__":
    main()