"""
MPRSS Module 7 - Sandbox Orchestration API
Ties the full pipeline together: APK upload -> package name extraction ->
install -> spawn+hook (behavioral log) -> network capture slice -> scoring
through Module 2 (static, port 5000) and Module 3 (anomaly, port 5001).

This is the "automated" version of everything we proved manually today:
  - Package name extraction:      androguard (same library Module 2 uses)
  - Install + spawn:               adb + frida.get_usb_device().spawn()
  - Behavioral hooking:            behavior_hooks_compiled.js (proven working
                                    against YouTube and F-Droid)
  - Network capture:               sliced from the emulator's CONTINUOUS
                                    -tcpdump capture, by timestamp range
                                    (see IMPORTANT note below)
  - Static + anomaly scoring:      delegated via HTTP to the existing
                                    Module 2 / Module 3 Flask apps, so this
                                    endpoint doesn't duplicate their logic

Run with: python module7_orchestrator.py   (default port 5002)

IMPORTANT - emulator setup requirement:
    The emulator must be running with a continuous tcpdump capture, e.g.:
        emulator -avd Pixel_6_Pro -no-snapshot -tcpdump C:\\path\\to\\continuous_capture.pcap
    Set CONTINUOUS_CAPTURE_PATH below to match that path. This endpoint
    slices each run's window out of that ongoing file rather than
    restarting the emulator per APK (which would kill the frida-server /
    installed-app state between runs).

IMPORTANT - root/SELinux requirement:
    Before calling this endpoint, run once per emulator session:
        adb root
        adb shell setenforce 0
        adb shell "/data/local/tmp/frida-server &"
    (Frida's spawn+attach will time out otherwise - see today's session notes.)
"""

import os
import re
import json
import time
import shutil
import tempfile
import subprocess

import requests
from flask import Flask, request, jsonify
from flask_cors import CORS

import frida
from scapy.utils import PcapReader, PcapWriter
from risk_scorer import compute_combined_risk
import family_predictor

from loguru import logger
import sys
logger.remove()
logger.add(sys.stderr, level="WARNING")

import logging
logging.getLogger('androguard').setLevel(logging.WARNING)
try:
    from androguard.core.apk import APK  # androguard 4.x layout
except ImportError:
    from androguard.core.bytecodes.apk import APK  # androguard 3.x layout

# ---------------------------------------------------------------------------
app = Flask(__name__)
CORS(app)

ADB_PATH = r"C:\Users\numl-\AppData\Local\Android\Sdk\platform-tools\adb.exe"
HOOK_SCRIPT_PATH = "behavior_hooks_compiled.js"
CONTINUOUS_CAPTURE_PATH = r"C:\Users\numl-\Downloads\continuous_capture.pcap"

MODULE2_URL = "http://localhost:5000/api/analyze/apk"
MODULE3_URL = "http://localhost:5001/api/analyze/flow"

RUN_DURATION_SECONDS = 120  # how long to monitor behavior + capture traffic

# Every run's network capture is KEPT here (one .pcap per analysis, named by package + timestamp)
# so the traffic can be opened later in Wireshark. Previously these were temp files deleted after scoring.
CAPTURES_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "captures")


# ---------------------------------------------------------------------------
def adb(*args):
    """Run an adb command, return (returncode, stdout, stderr)."""
    result = subprocess.run([ADB_PATH] + list(args), capture_output=True, text=True)
    return result.returncode, result.stdout, result.stderr


def preflight_check():
    """
    Verify the emulator is actually ready for Frida BEFORE installing anything,
    and fail with a message that says exactly which setup step is missing.
    Without this, a missing step (e.g. frida-server not running, SELinux back to
    Enforcing after an emulator restart) only surfaces ~1-2 minutes into the run
    as a vague "frida.TimedOutError: timeout was reached".
    """
    code, out, err = adb('get-state')
    if code != 0 or out.strip() != 'device':
        raise RuntimeError("No emulator connected to adb. Start the emulator, then check 'adb devices'.")

    code, out, err = adb('shell', 'getprop', 'sys.boot_completed')
    if out.strip() != '1':
        raise RuntimeError("Emulator is not fully booted (or its system crashed). Cold-boot it and retry.")

    code, out, err = adb('shell', 'id')
    if 'uid=0' not in out:
        raise RuntimeError("adb is not running as root. Run: adb root")

    code, out, err = adb('shell', 'getenforce')
    if out.strip().lower() != 'permissive':
        raise RuntimeError("SELinux is still Enforcing, so Frida will time out. Run: adb shell setenforce 0")

    code, out, err = adb('shell', 'pidof', 'frida-server')
    if code != 0 or not out.strip():
        raise RuntimeError(
            "frida-server is not running on the emulator. Start it with: "
            "adb shell \"nohup /data/local/tmp/frida-server >/dev/null 2>&1 &\""
        )


def extract_package_name(apk_path):
    """
    Extract the package identifier from an APK using androguard - the same
    library Module 2's apk_static_parser.py already depends on. Uses the
    lightweight APK() manifest parser (not full AnalyzeAPK) since we only
    need the package name here, not the DEX/method analysis.
    """
    a = APK(apk_path)
    package_name = a.get_package()
    if not package_name:
        raise ValueError("Could not extract package name from APK manifest")
    return package_name


def install_apk(apk_path):
    code, out, err = adb('install', '-r', apk_path)
    if code != 0 or 'Success' not in out:
        raise RuntimeError(f"adb install failed: {out}\n{err}")


def run_behavioral_capture(package_name, duration):
    """
    Spawn the app fresh (paused), attach Frida, load the hook script,
    resume, monitor for `duration` seconds, collect all reported events.
    Mirrors frida_behavior_logger.py's proven approach.

    Returns (events: list[dict], run_start_epoch: float, run_end_epoch: float)
    """
    if not os.path.exists(HOOK_SCRIPT_PATH):
        raise FileNotFoundError(
            f"Hook script not found at '{HOOK_SCRIPT_PATH}'. Run "
            f"'npx frida-compile behavior_hooks.js -o behavior_hooks_compiled.js' first."
        )

    _FridaTimeout = getattr(frida, "TimedOutError", TimeoutError)
    try:
        device = frida.get_usb_device(timeout=10)
    except _FridaTimeout:
        raise RuntimeError(
            "Frida timed out finding the emulator. Check that frida-server is running, adb is root, "
            "SELinux is Permissive, and the frida versions on PC and emulator match."
        )

    events = []

    def on_message(message, data):
        if message['type'] == 'send':
            events.append(message['payload'])

    try:
        pid = device.spawn([package_name])
        session = device.attach(pid)
    except _FridaTimeout:
        raise RuntimeError(
            "Frida timed out spawning/attaching to the app. Check that frida-server is running, adb is root, "
            "SELinux is Permissive, and the frida versions on PC and emulator match."
        )

    with open(HOOK_SCRIPT_PATH, 'r', encoding='utf-8') as f:
        script_source = f.read()

    script = session.create_script(script_source, runtime='v8')
    script.on('message', on_message)
    script.load()

    run_start = time.time()
    device.resume(pid)

    time.sleep(duration)

    run_end = time.time()
    session.detach()

    return events, run_start, run_end


def get_wifi_interface_index():
    """Look up the current Wi-Fi interface number from tshark -D output,
    since Windows can reassign interface indices across reboots."""
    tshark_path = r"C:\Program Files\Wireshark\tshark.exe"
    result = subprocess.run([tshark_path, "-D"], capture_output=True, text=True)
    for line in result.stdout.splitlines():
        if "(Wi-Fi)" in line:
            return line.split(".")[0].strip()
    raise RuntimeError("Could not find Wi-Fi interface in tshark -D output")


def capture_traffic_live(duration, output_path):
    """
    Capture traffic directly via tshark for the duration of the run,
    instead of relying on the emulator's -tcpdump flag (proven unreliable
    today - it silently dies after restarts). Looks up the Wi-Fi
    interface fresh each call, since Windows can reassign interface
    index numbers across reboots.
    """
    tshark_path = r"C:\Program Files\Wireshark\tshark.exe"
    iface = get_wifi_interface_index()
    proc = subprocess.Popen([
        tshark_path, "-i", iface, "-a", f"duration:{duration}",
        "-w", output_path
    ])
    return proc

def slice_capture_by_time(run_start, run_end, output_path, padding_seconds=2):
    """
    Extract only the packets that fall within [run_start, run_end] (with a
    small padding buffer) from the emulator's continuous tcpdump capture.
    This avoids needing to restart the emulator (and lose installed apps /
    frida-server state) for every single APK analyzed.
    """
    if not os.path.exists(CONTINUOUS_CAPTURE_PATH):
        raise FileNotFoundError(
            f"Continuous capture file not found at '{CONTINUOUS_CAPTURE_PATH}'. "
            f"Make sure the emulator was launched with -tcpdump pointing here."
        )

    lo = run_start - padding_seconds
    hi = run_end + padding_seconds

    writer = PcapWriter(output_path, append=False, sync=True)
    matched = 0
    with PcapReader(CONTINUOUS_CAPTURE_PATH) as reader:
        for pkt in reader:
            if hasattr(pkt, 'time') and lo <= float(pkt.time) <= hi:
                writer.write(pkt)
                matched += 1
    writer.close()

    return matched


def score_static(apk_path):
    """Delegate to Module 2's existing Flask API."""
    with open(apk_path, 'rb') as f:
        resp = requests.post(MODULE2_URL, files={'apk': f}, timeout=360)
    if resp.status_code != 200:
        return {'error': f'Module 2 returned {resp.status_code}: {resp.text}'}
    return resp.json()


def score_flow(pcap_path):
    """Delegate to Module 3's existing Flask API."""
    if not os.path.exists(pcap_path) or os.path.getsize(pcap_path) == 0:
        return {'anomaly_score': None, 'note': 'No packets captured for this run'}
    with open(pcap_path, 'rb') as f:
        resp = requests.post(MODULE3_URL, files={'pcap': f}, timeout=60)
    if resp.status_code != 200:
        return {'error': f'Module 3 returned {resp.status_code}: {resp.text}'}
    return resp.json()


def summarize_behavior(events):
    """Simple aggregate summary of the behavioral log for the response."""
    by_category = {}
    for e in events:
        cat = e.get('category', 'unknown')
        by_category[cat] = by_category.get(cat, 0) + 1
    return {
        'total_events': len(events),
        'events_by_category': by_category,
    }


# ---------------------------------------------------------------------------
@app.route('/api/health', methods=['GET'])
def health():
    return jsonify({'status': 'ok'})


@app.route('/api/analyze/full', methods=['POST'])
def analyze_full():
    """
    Full orchestrated analysis: upload an APK, get back static score,
    anomaly score, and behavioral event summary in one response.

    Request: multipart/form-data with a file field named 'apk'
    """
    if 'apk' not in request.files:
        return jsonify({'error': "Request must include a file field named 'apk'"}), 400

    file = request.files['apk']
    if file.filename == '':
        return jsonify({'error': 'No file selected'}), 400

    tmp_apk = None
    tmp_pcap = None
    try:
        with tempfile.NamedTemporaryFile(delete=False, suffix='.apk') as tmp:
            file.save(tmp.name)
            tmp_apk = tmp.name

        # Step 0: make sure the emulator + frida-server are really ready (fail fast, clear message)
        preflight_check()

        # Step 1: extract package name
        package_name = extract_package_name(tmp_apk)

        # Step 2: install
        install_apk(tmp_apk)

        # Step 3+4: start live packet capture, then run behavioral hooks concurrently
        os.makedirs(CAPTURES_DIR, exist_ok=True)
        timestamp = time.strftime('%Y%m%d_%H%M%S')
        tmp_pcap = os.path.join(CAPTURES_DIR, f'{package_name}_{timestamp}.pcap')  # kept permanently
        tshark_proc = capture_traffic_live(RUN_DURATION_SECONDS, tmp_pcap)
        events, run_start, run_end = run_behavioral_capture(package_name, RUN_DURATION_SECONDS)
        tshark_proc.wait(timeout=15)
        packet_count = None  # scored directly by Module 3 from the file instead

        # Step 5: score static (Module 2) and flow (Module 3) in parallel-ish (sequential is fine here)
        static_result = score_static(tmp_apk)
        flow_result = score_flow(tmp_pcap)
        behavior_summary = summarize_behavior(events)

        # Step 6: Module 5 family classification - runs entirely in-process
        # (no HTTP call, unlike Module 2/3), using the models loaded once at
        # startup. Depends on Module 2's response including 'active_features'
        # (the list apk_static_parser.py produces) and Module 3's response
        # including 'flow_features' (added above in app.py) - if either key
        # is absent, this degrades gracefully to an empty/unclassified result
        # rather than failing the whole analysis.
        family_static = family_predictor.predict_static_family(
            static_result.get('active_features', [])
        )
        family_flow = family_predictor.predict_flow_family(
            flow_result.get('flow_features', [])
        )

        combined_risk = compute_combined_risk(
        static_score=static_result.get('static_score'),
        anomaly_score=flow_result.get('anomaly_score'),
        behavioral_score=None,  # Module 1 not trained yet
        static_confidence=static_result.get('confidence'),
        )

        return jsonify({
            'package_name': package_name,
            'run_duration_seconds': RUN_DURATION_SECONDS,
            'packets_captured': packet_count,
            'pcap_saved_path': tmp_pcap,
            'combined_risk': combined_risk,
            'static_analysis': static_result,
            'anomaly_analysis': flow_result,
            'behavioral_summary': behavior_summary,
            'behavioral_events': events,
            'family_classification': {
                'static': family_static,
                'flow': family_flow,
            },
        })
    except Exception as e:
        import traceback
        traceback.print_exc()
        return jsonify({'error': str(e)}), 500

    finally:
        for path in (tmp_apk,):  # pcap is intentionally kept in CAPTURES_DIR
            if path and os.path.exists(path):
                try:
                    os.remove(path)
                except PermissionError:
                    pass  # Windows sometimes briefly locks files after a subprocess closes them; not fatal


if __name__ == '__main__':
    print("MPRSS Module 7 Orchestrator starting on port 5002...")
    print(f"Continuous capture path: {CONTINUOUS_CAPTURE_PATH}")
    print(f"Hook script: {HOOK_SCRIPT_PATH}")
    print("Reminder: run 'adb root && adb shell setenforce 0' once per emulator session first.")
    family_predictor.load_family_models('models')  # Module 5 - static+flow K-Means artifacts
    app.run(debug=True, port=5002)