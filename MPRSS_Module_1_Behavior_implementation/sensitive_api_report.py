"""
sensitive_api_report.py - pick Module 1's vocabulary by MEANING, then check it against the data.

Usage (from the MPRSS_Module1_Behavior folder):
    python sensitive_api_report.py

Why not just take the statistically strongest APIs (select_vocab.py)?
  * The top of that list was one chain of Parcel/Binder calls that always occur
    together - one library or one group of similar apps, not many behaviors.
  * Those calls happen constantly inside Android itself. The training traces
    only record calls made by the APP's own code, but a Frida hook on the
    framework method sees EVERY caller. Hooking them would flood the event
    stream and slow the app, and live data would not match the training data.
  * Many candidates leaned opposite ways in 2018 vs 2019.

So: start from security-relevant API categories (telephony, network, process
execution, dynamic code loading, crypto, package info, files, ...). These are
rarely called by the system itself, so Frida hooks stay cheap and match the
training traces. Then use the data only to see which ones actually occur.
"""

import os
import csv
import json
import argparse

import numpy as np

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

CATEGORIES = {
    "telephony / SMS": ("android.telephony.",),
    "location": ("android.location.",),
    "network": ("java.net.", "org.apache.http.", "javax.net.ssl.", "android.net.", "android.net.wifi."),
    "process execution": ("java.lang.Runtime.", "java.lang.ProcessBuilder.", "java.lang.Process."),
    "dynamic loading / reflection": ("dalvik.system.", "java.lang.ClassLoader.", "java.lang.Class.forName",
                                     "java.lang.Class.getMethod", "java.lang.Class.getDeclaredMethod",
                                     "java.lang.reflect.", "java.lang.System.load"),
    "crypto / encoding": ("javax.crypto.", "java.security.", "android.util.Base64."),
    "package / system info": ("android.content.pm.PackageManager.", "android.app.ActivityManager.",
                              "android.os.Process.", "android.provider."),
    "files": ("java.io.File.", "java.io.FileOutputStream.", "java.io.FileInputStream.",
              "java.io.RandomAccessFile.", "android.os.Environment."),
    "content providers": ("android.content.ContentResolver.",),
    "components / broadcasts": ("android.content.ContextWrapper.startService", "android.content.ContextWrapper.sendBroadcast",
                                "android.content.ContextWrapper.bindService", "android.content.ContextWrapper.startActivity",
                                "android.content.ContextWrapper.registerReceiver", "android.content.ContextWrapper.getSystemService",
                                "android.content.Context.startService", "android.content.Context.sendBroadcast",
                                "android.content.Context.bindService", "android.content.Context.registerReceiver",
                                "android.content.Context.getSystemService", "android.app.AlarmManager.",
                                "android.app.NotificationManager.", "android.content.BroadcastReceiver.abortBroadcast"),
    "camera / audio": ("android.hardware.", "android.media.MediaRecorder.", "android.media.AudioRecord."),
    "accounts / device admin": ("android.accounts.", "android.app.admin."),
    "webview": ("android.webkit.WebView.", "android.webkit.WebSettings."),
}
EXCLUDE_PREFIXES = ("android.net.Uri",)   # string parsing helper, not a behavior


def category_of(api):
    if api.startswith(EXCLUDE_PREFIXES):
        return None
    for cat, prefixes in CATEGORIES.items():
        if api.startswith(prefixes):
            return cat
    return None


def load_cache(cache_dir):
    z = np.load(os.path.join(cache_dir, "api_cache.npz"))
    with open(os.path.join(cache_dir, "api_vocab.json"), encoding="utf-8") as f:
        vocab = json.load(f)
    return z["tokens"], z["offsets"], z["labels"], vocab


def year_stats(cache_dir, n):
    """Per API: (% benign apps, % malware apps, avg calls per app) using only apps with >= n framework
    calls and only their first n calls. Also returns sensitive-call counts per app for the sparsity check."""
    tokens, offsets, labels, vocab = load_cache(cache_dir)
    lengths = offsets[:, 1] - offsets[:, 0]
    keep = np.where(lengths >= n)[0]
    v = len(vocab)
    is_sensitive = np.array([category_of(a) is not None for a in vocab])
    present = {0: np.zeros(v), 1: np.zeros(v)}
    occ = np.zeros(v)
    n_apps = {0: 0, 1: 0}
    per_app = {0: [], 1: []}
    for i in keep:
        s = offsets[i, 0]
        seq = tokens[s:s + n]
        counts = np.bincount(seq, minlength=v)
        lab = int(labels[i])
        present[lab] += counts > 0
        occ += counts
        n_apps[lab] += 1
        per_app[lab].append(int(is_sensitive[seq].sum()))
    total = max(n_apps[0] + n_apps[1], 1)
    stats = {a: (100.0 * present[0][i] / max(n_apps[0], 1), 100.0 * present[1][i] / max(n_apps[1], 1), occ[i] / total)
             for i, a in enumerate(vocab)}
    return stats, n_apps, per_app


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cache2018", default="logs2018")
    ap.add_argument("--cache2019", default="logs")
    ap.add_argument("--n", type=int, default=1000)
    ap.add_argument("--min-support", type=float, default=2.0, help="show APIs used by >= this %% of apps (any class/year)")
    ap.add_argument("--per-category", type=int, default=12)
    ap.add_argument("--out", default=os.path.join("logs", "sensitive_candidates.csv"))
    args = ap.parse_args()

    s18, n18, pa18 = year_stats(args.cache2018, args.n)
    s19, n19, pa19 = year_stats(args.cache2019, args.n)
    print(f"Apps with >= {args.n} framework calls (only first {args.n} calls used):")
    print(f"   2018: benign={n18[0]}  malware={n18[1]}      2019: benign={n19[0]}  malware={n19[1]}")

    rows = []
    for api in sorted(set(s18) | set(s19)):
        cat = category_of(api)
        if cat is None:
            continue
        b18, m18, o18 = s18.get(api, (0.0, 0.0, 0.0))
        b19, m19, o19 = s19.get(api, (0.0, 0.0, 0.0))
        if max(b18, m18, b19, m19) < args.min_support:
            continue
        w18, w19 = n18[0] + n18[1], n19[0] + n19[1]
        rate = (o18 * w18 + o19 * w19) / (w18 + w19)
        d18, d19 = m18 - b18, m19 - b19
        same_dir = (d18 > 0 and d19 > 0) or (d18 < 0 and d19 < 0)
        rows.append((cat, api, b18, m18, b19, m19, rate, same_dir))

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    with open(args.out, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["category", "api", "benign_2018_pct", "malware_2018_pct", "benign_2019_pct", "malware_2019_pct",
                    "calls_per_app", "same_direction_both_years", "already_hooked"])
        for r in sorted(rows, key=lambda r: (r[0], -max(r[2:6]))):
            w.writerow([r[0], r[1], f"{r[2]:.1f}", f"{r[3]:.1f}", f"{r[4]:.1f}", f"{r[5]:.1f}", f"{r[6]:.2f}", r[7], r[1] in FRIDA_HOOKED])

    print(f"\nSensitive APIs seen in >= {args.min_support}% of apps (% of apps calling each; * = already hooked; "
          f"HOT = >10 calls/app, costly to hook; -- = leans opposite ways in the two years)")
    for cat in CATEGORIES:
        cat_rows = sorted([r for r in rows if r[0] == cat], key=lambda r: -max(r[2:6]))[:args.per_category]
        if not cat_rows:
            print(f"\n[{cat}]  (nothing reaches {args.min_support}% support)")
            continue
        print(f"\n[{cat}]   {'ben18':>5}{'mal18':>6}{'ben19':>6}{'mal19':>6}{'calls/app':>10}")
        for r in cat_rows:
            flags = ("*" if r[1] in FRIDA_HOOKED else " ") + ("HOT " if r[6] > 10 else "    ") + ("" if r[7] else "--")
            print(f"  {r[1][:58]:<58}{r[2]:>5.0f}{r[3]:>6.0f}{r[4]:>6.0f}{r[5]:>6.0f}{r[6]:>10.2f}  {flags}")

    print("\nHOW MANY sensitive-API calls does an app make within its first "
          f"{args.n} framework calls? (all categories above, any support)")
    print(f"   {'group':<16}{'apps with 0':>12}{'median':>8}{'p90':>6}")
    for name, per_app in (("2018 benign", pa18[0]), ("2018 malware", pa18[1]), ("2019 benign", pa19[0]), ("2019 malware", pa19[1])):
        a = np.array(per_app)
        print(f"   {name:<16}{100 * np.mean(a == 0):>11.0f}%{np.median(a):>8.0f}{np.percentile(a, 90):>6.0f}")

    print(f"\nFull list saved to {args.out}")
    print("Please send me everything printed above (as text).")


if __name__ == "__main__":
    main()