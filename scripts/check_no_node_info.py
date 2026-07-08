#!/usr/bin/env python3
"""Pre-commit guard enforcing the AGENTS.md rule:

    "Absolutely no network, node, or firmware info should ever be checked in."

Scans the *staged content* of the files passed on argv for things that look like
network / node / firmware identifiers (IP/MAC addresses, GPU UUIDs, PCI bus IDs,
serials, vbios/bmc/ipmi/firmware tokens, hostnames) and fails the commit if any
are found.

Stdlib only, no third-party deps. Deliberately allows the porting target GPU
model strings (MI350X / gfx950 / Instinct / ROCm versions), which the owner has
ruled acceptable to check in.
"""

from __future__ import annotations

import re
import sys

# Substrings (case-insensitive) that are explicitly allowed and must never be
# flagged. These are the porting target GPU model / stack identifiers.
ALLOWLIST_SUBSTRINGS = (
    "mi350x",
    "mi300x",
    "gfx950",
    "gfx942",
    "instinct",
    "rocm",
    "hip",
    "cdna",
)

# Lines matching these (case-insensitive) are skipped entirely. Keeps common
# false positives (version strings, semver-ish tokens) from tripping the IP
# regex, while still catching real addresses elsewhere on other lines.
ALLOW_LINE_PATTERNS = (
    re.compile(r"\brocm[\s/_-]*\d", re.IGNORECASE),
    re.compile(r"\bpython\s*\d", re.IGNORECASE),
)

# Detection patterns. Each entry: (label, compiled regex).
PATTERNS: list[tuple[str, re.Pattern[str]]] = [
    (
        "MAC address",
        re.compile(r"\b(?:[0-9a-fA-F]{2}:){5}[0-9a-fA-F]{2}\b"),
    ),
    (
        "GPU UUID",
        re.compile(r"\bGPU-[0-9a-fA-F]{8}(?:-[0-9a-fA-F]{4}){3}-[0-9a-fA-F]{12}\b"),
    ),
    (
        "GPU UUID (short)",
        re.compile(r"\bGPU-[0-9a-fA-F]{6,}\b"),
    ),
    (
        "PCI bus id",
        re.compile(r"\b[0-9a-fA-F]{4}:[0-9a-fA-F]{2}:[0-9a-fA-F]{2}\.[0-9a-fA-F]\b"),
    ),
    (
        "IPv6 address",
        re.compile(r"(?<![:.\w])(?:[0-9a-fA-F]{1,4}:){2,7}[0-9a-fA-F]{1,4}(?![:.\w])"),
    ),
    (
        "firmware/node token",
        re.compile(
            r"\b(?:vbios|firmware|serial(?:\s*number|no)?|bmc|ipmi|idrac|ilo|"
            r"hostname)\b\s*[:=]",
            re.IGNORECASE,
        ),
    ),
]

# IPv4 handled specially so we can drop obvious version numbers / non-addresses.
IPV4_RE = re.compile(r"\b(\d{1,3})\.(\d{1,3})\.(\d{1,3})\.(\d{1,3})\b")


def _is_ipv4(match: re.Match[str]) -> bool:
    return all(0 <= int(g) <= 255 for g in match.groups())


def _line_allowed(line: str) -> bool:
    low = line.lower()
    if any(tok in low for tok in ALLOWLIST_SUBSTRINGS):
        return True
    return any(p.search(line) for p in ALLOW_LINE_PATTERNS)


def scan_text(text: str) -> list[str]:
    findings: list[str] = []
    for lineno, line in enumerate(text.splitlines(), start=1):
        if _line_allowed(line):
            continue
        stripped = line.strip()
        for label, pat in PATTERNS:
            for m in pat.finditer(line):
                findings.append(f"  line {lineno}: {label}: {m.group(0)!r}")
        for m in IPV4_RE.finditer(line):
            if _is_ipv4(m):
                findings.append(f"  line {lineno}: IPv4 address: {m.group(0)!r}")
        del stripped
    return findings


def read_staged(path: str) -> str | None:
    """Read the staged (index) version of a file, matching what will commit."""
    import subprocess

    try:
        out = subprocess.run(
            ["git", "show", f":{path}"],
            capture_output=True,
            check=True,
        )
    except subprocess.CalledProcessError:
        return None
    try:
        return out.stdout.decode("utf-8")
    except UnicodeDecodeError:
        # Binary file; nothing textual to scan.
        return None


def main(argv: list[str]) -> int:
    files = argv[1:]
    failed = False
    for path in files:
        content = read_staged(path)
        if content is None:
            continue
        findings = scan_text(content)
        if findings:
            failed = True
            print(f"[no-node-info] potential network/node/firmware info in {path}:")
            print("\n".join(findings))
    if failed:
        print()
        print(
            "[no-node-info] Commit blocked: remove the above network/node/firmware "
            "info (see AGENTS.md). Allowed GPU model strings (MI350X/gfx950/"
            "Instinct/ROCm) are not flagged."
        )
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
