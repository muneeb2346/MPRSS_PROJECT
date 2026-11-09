"""
peek_androct.py - look inside an AndroCT trace archive WITHOUT extracting it.

Usage:
    python peek_androct.py "C:\\path\\to\\trace-benign-2018.tar.gz"

Why this exists: Module 1's LSTM needs to be designed around what the trace
lines ACTUALLY look like (which fields exist, how long a trace is, how many
apps there are). We look first, then build - the same way we checked the
flow dataset's columns before deciding it could not support sequence modeling.

Safe to run: the archive holds text call-trace logs, not APK/executable
files, and this script only READS text from it.
"""

import os
import sys
import tarfile
import statistics
import collections


def count_lines(fileobj, chunk_size=1 << 20):
    """Count newline characters in a file object, reading in chunks."""
    total = 0
    while True:
        chunk = fileobj.read(chunk_size)
        if not chunk:
            return total
        total += chunk.count(b"\n")


def main(path, n_preview_files=2, n_preview_lines=15):
    print(f"\nFile: {path}")
    print(f"Compressed size: {os.path.getsize(path) / 1e6:.1f} MB\n")

    sizes = []
    ext_counts = collections.Counter()
    names = []
    previews = []

    with tarfile.open(path, "r:gz") as tar:
        for member in tar:
            if not member.isfile():
                continue
            names.append(member.name)
            sizes.append(member.size)
            ext_counts[os.path.splitext(member.name)[1].lower() or "(no extension)"] += 1

            # Preview the first few non-empty files: first lines + total line count
            if len(previews) < n_preview_files and member.size > 0:
                lines = []
                with tar.extractfile(member) as f:
                    for i, raw in enumerate(f):
                        if i >= n_preview_lines:
                            break
                        lines.append(raw.decode("utf-8", "replace").rstrip())
                with tar.extractfile(member) as f:
                    total_lines = count_lines(f)
                previews.append((member.name, member.size, total_lines, lines))

    if not sizes:
        print("No files found inside the archive.")
        return

    print("=" * 60)
    print("ARCHIVE CONTENTS")
    print("=" * 60)
    print(f"Files inside:            {len(names)}")
    print(f"File types:              {dict(ext_counts)}")
    print(f"Uncompressed size (KB):  min={min(sizes)/1e3:.1f}  "
          f"median={statistics.median(sizes)/1e3:.1f}  "
          f"max={max(sizes)/1e3:.1f}")
    print("First 5 file names:")
    for n in names[:5]:
        print(f"   {n}")

    for name, size, total_lines, lines in previews:
        print("\n" + "=" * 60)
        print(f"PREVIEW: {name}")
        print(f"   size: {size/1e3:.1f} KB   total lines (= sequence length): {total_lines}")
        print("=" * 60)
        for ln in lines:
            print(ln[:200])


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print(__doc__)
        sys.exit(1)
    main(sys.argv[1])