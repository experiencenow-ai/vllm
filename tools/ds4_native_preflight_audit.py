#!/usr/bin/env python3
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def check(label: str, ok: bool) -> bool:
    print(("PASS" if ok else "FAIL") + ": " + label)
    return ok


def main() -> int:
    guard = (ROOT / "tools/ds4_200g_guard.sh").read_text()
    ok = True
    ok &= check(
        "DS4 native preflight has a separate active layout probe gate",
        "DS4_NATIVE_PREFLIGHT_ACTIVE_LAYOUT" in guard
        and "active_layout_probe" in guard,
    )
    ok &= check(
        "inactive native preflight skips the SF layout CUDA probe",
        "--skip-active-layout-probe" in guard
        and 'if [[ "$active_layout_probe" != "1" ]]' in guard,
    )
    ok &= check(
        "active kernel probe remains independently controlled",
        'if [[ "$active_probe" == "1" ]]' in guard
        and "--active-kernel-probe" in guard,
    )
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
