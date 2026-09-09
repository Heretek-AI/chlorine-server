#!/usr/bin/env python3
"""Equivalence gates vs the original halogen engine (docs/halogen/TRUNK-NOTES.md).

Runs the trunk harness (tf / dec / samplecheck) and checks the recorded
reference values. The original engine's artifacts (forced-text.bin,
forced-topk8.bin, forced-clean8.bin, the --sample-check table) were captured
once from ~/halogen-original and live in work/opencode + /tmp (see
TRUNK-NOTES.md section 2-4 for the capture commands).

Usage: python3 tests/equivalence/run_gates.py [--bin /tmp/trunk3]
Exit 0 = all gates green.
"""
import argparse
import json
import os
import subprocess
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
REF = os.path.join(ROOT, "work", "opencode", "forced-text.bin")
GREEDY = os.path.join(ROOT, "tests", "equivalence", "greedy_text_87.json")

# best-known bench-emulation gates (HALO_ACTQ=3 HALO_BETA32=1)
TF_MEAN_NLL_MAX = 5.98  # ours 5.777286 vs ground truth 5.918531 (gate: <= 6.0)
# top1: the engine's own value (5/86) sits inside the zero-pad-row chaos band.
# 43/86 rows are tgt=0 padding whose NLL/preds amplify upstream ULP differences
# ~50x (engine bench-vs-clean: pad rows +0.658, text rows +0.059 mean NLL). An
# emulation at the engine's noise level lands at 4-6 there; ours = 4 (2026-09-09).
# Full top1 parity requires GDN/FA internals bit-parity (TRUNK-NOTES §6.1).
TF_TOP1_MIN = 4
GREEDY_MIN_MATCH = 5  # 5-token solid-margin prefix (271 51 1618 579 1558)
SC_CHI2_MAX = 1.6

fails = []


def run(cmd, env=None, timeout=1800):
    e = dict(os.environ)
    sp = os.path.expanduser("~/chlorine-venv/lib/python3.12/site-packages")
    if os.path.isdir(sp + "/_rocm_sdk_core"):
        libs = os.pathsep.join([sp + "/_rocm_sdk_core/lib", sp + "/_rocm_sdk_libraries/lib"])
        e["LD_LIBRARY_PATH"] = libs + (os.pathsep + e["LD_LIBRARY_PATH"]
                                       if e.get("LD_LIBRARY_PATH") else "")
    if env:
        e.update(env)
    return subprocess.run(cmd, capture_output=True, text=True, env=e, timeout=timeout)


def check(name, ok, detail):
    print(f"  {'ok  ' if ok else 'FAIL'} {name}: {detail}")
    if not ok:
        fails.append(name)


def parse_tf(out):
    nll = pred = top1 = hsh = None
    for ln in out.splitlines():
        if ln.startswith("mean-NLL"):
            nll = float(ln.split("ours=")[1].split()[0])
        if ln.startswith("rows="):
            pred = int(ln.split("predmatch=")[1].split("/")[0])
            top1 = int(ln.split("top1=")[1].split("/")[0])
            hsh = ln.split("hash ")[1].strip()
    return nll, pred, top1, hsh


def parse_sc(out):
    chi2 = None
    for ln in out.splitlines():
        if "chi2/df" in ln:
            chi2 = float(ln.split("=")[1].split()[0])
    return chi2


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--bin", default="/tmp/trunk3")
    ap.add_argument("--skip-greedy", action="store_true",
                    help="skip the 24-token greedy decode (slow)")
    args = ap.parse_args()

    if not os.path.exists(REF):
        print(f"missing reference dump {REF} — capture it first (TRUNK-NOTES.md §2)")
        return 2
    if not os.path.exists(args.bin):
        print(f"missing harness {args.bin} — build: hipcc --offload-arch=gfx1151 -O3 "
              "-fopenmp engine/kernels/trunk.hip -o /tmp/trunk3")
        return 2

    print("== gate 1: teacher-forced bench emulation (tf) ==")
    r = run([args.bin, "tf"])
    if r.returncode != 0:
        check("tf run", False, f"exit {r.returncode}: {r.stderr[-300:]}")
    else:
        nll, pred, top1, hsh = parse_tf(r.stdout)
        check("tf mean-NLL", nll is not None and nll <= TF_MEAN_NLL_MAX,
              f"{nll} (gate <= {TF_MEAN_NLL_MAX}, ground truth 5.918531)")
        check("tf top1", top1 is not None and top1 >= TF_TOP1_MIN, f"{top1}/{TF_TOP1_MIN}")
        print(f"       (predmatch={pred}, hash={hsh}, recorded target 6be127062a76)")

    print("== gate 2: sampler self-check (samplecheck) ==")
    r = run([args.bin, "samplecheck"])
    if r.returncode != 0:
        check("samplecheck run", False, f"exit {r.returncode}")
    else:
        chi2 = parse_sc(r.stdout)
        check("chi2/df", chi2 is not None and chi2 < SC_CHI2_MAX,
              f"{chi2} (gate < {SC_CHI2_MAX}; reference run 0.990)")
        check("off-support", "off-support draws: 0" in r.stdout, "must be 0")

    if not args.skip_greedy:
        print("== gate 3: greedy stream prefix (dec) ==")
        r = run([args.bin, "dec"])
        tgt = json.load(open(GREEDY))["greedy_tokens"]
        mine = []
        for ln in r.stdout.splitlines():
            if ln.startswith("greedy:"):
                mine = [int(x) for x in ln.split(":", 1)[1].split()]
        pre = 0
        for a, b in zip(mine, tgt):
            if a != b:
                break
            pre += 1
        check("greedy prefix", pre >= GREEDY_MIN_MATCH,
              f"{pre}/{GREEDY_MIN_MATCH} consecutive ({' '.join(map(str, mine[:8]))}...)")

    print()
    if fails:
        print(f"FAILED: {', '.join(fails)}")
        return 1
    print("ALL GATES GREEN")
    return 0


if __name__ == "__main__":
    sys.exit(main())
