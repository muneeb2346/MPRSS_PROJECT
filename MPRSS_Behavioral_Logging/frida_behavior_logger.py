"""
MPRSS Module 1 - Behavioral Logger
Attaches Frida to a running app on the connected Android device/emulator,
loads behavior_hooks.js, and writes every reported event to a timestamped
JSONL log file. This is the data-collection half of Module 1 - each run
against a real APK produces one log file; accumulate many runs (benign
and malicious samples) before training the RNN/LSTM.

Usage:
    python frida_behavior_logger.py "1Win"
    python frida_behavior_logger.py "1Win" --duration 60
    python frida_behavior_logger.py "1Win" --output logs/1win_run1.jsonl

Requires: pip install frida frida-tools
Requires: frida-server already running on the device (adb shell,
          confirmed earlier via `frida-ps -U`)
"""

import argparse
import json
import os
import sys
import time
from datetime import datetime

import frida


def build_output_path(app_name, explicit_path):
    if explicit_path:
        out_dir = os.path.dirname(explicit_path)
        if out_dir:
            os.makedirs(out_dir, exist_ok=True)
        return explicit_path

    os.makedirs('logs', exist_ok=True)
    safe_name = ''.join(c if c.isalnum() else '_' for c in app_name)
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    return os.path.join('logs', f'{safe_name}_{timestamp}.jsonl')


def main():
    parser = argparse.ArgumentParser(description='MPRSS Module 1 behavioral logger')
    parser.add_argument('app_name', help="App process name as shown by 'frida-ps -U' (e.g. '1Win')")
    parser.add_argument('--duration', type=int, default=60,
                         help='How many seconds to monitor before detaching (default: 60)')
    parser.add_argument('--output', default=None,
                         help='Output JSONL path (default: logs/<app>_<timestamp>.jsonl)')
    parser.add_argument('--script', default='behavior_hooks.js',
                         help='Path to the Frida JS agent (default: behavior_hooks.js)')
    args = parser.parse_args()

    if not os.path.exists(args.script):
        print(f"ERROR: hook script not found at '{args.script}'. "
              f"Make sure behavior_hooks.js is in the same folder, or pass --script.")
        sys.exit(1)

    output_path = build_output_path(args.app_name, args.output)
    event_count = 0

    print(f"{'='*60}")
    print(f"MPRSS Module 1 - Behavioral Logger")
    print(f"{'='*60}")
    print(f"Target app:    {args.app_name}")
    print(f"Duration:      {args.duration}s")
    print(f"Output log:    {output_path}")
    print(f"{'='*60}\n")

    try:
        device = frida.get_usb_device(timeout=10)
    except frida.TimeoutError:
        print("ERROR: No USB/adb device found. Is the emulator running and "
              "adb connected? (check `adb devices`)")
        sys.exit(1)

    try:
        pid = device.spawn([args.app_name])
        session = device.attach(pid)
        print(f"Attached to '{args.app_name}' successfully.")
    except frida.ProcessNotFoundError:
        print(f"ERROR: Process '{args.app_name}' not found. Run `frida-ps -U` "
              f"to see exact process names, and make sure the app is running.")
        sys.exit(1)

    with open(args.script, 'r', encoding='utf-8') as f:
        script_source = f.read()

    script = session.create_script(script_source, runtime='v8')

    log_file = open(output_path, 'w', encoding='utf-8')

    def on_message(message, data):
        nonlocal event_count
        if message['type'] == 'send':
            payload = message['payload']
            payload['logged_at'] = datetime.now().isoformat()
            log_file.write(json.dumps(payload) + '\n')
            log_file.flush()
            event_count += 1
            print(f"[{event_count:04d}] {payload['category']:15s} "
                  f"{payload['api']:35s} {payload['detail'][:60]}")
        elif message['type'] == 'error':
            print(f"[SCRIPT ERROR] {message.get('description', message)}")
        elif message['type'] == 'log':
            print(f"[AGENT LOG] {message.get('payload', message)}")

    script.on('message', on_message)
    script.load()
    time.sleep(0.5)
    device.resume(pid)

    print(f"\nMonitoring for {args.duration} seconds... "
          f"(interact with the app now to generate events)\n")

    try:
        time.sleep(args.duration)
    except KeyboardInterrupt:
        print("\nInterrupted early by user.")

    session.detach()
    log_file.close()

    print(f"\n{'='*60}")
    print(f"Done. {event_count} events logged to: {os.path.abspath(output_path)}")
    print(f"{'='*60}")


if __name__ == '__main__':
    main()