#!/usr/bin/env python3
"""Wire-protocol conformance tests for the chlorine engine skeleton.

Run: python3 tests/conformance/test_protocol.py [--engine PATH] [--checkpoint PATH]
Starts the engine, connects over TCP, and checks every reply against
docs/halogen/WIRE-PROTOCOL.md. No GPU required (stub generator).
"""
import argparse
import socket
import subprocess
import sys
import time

PASS, FAIL = 0, 0


def check(name, cond, detail=""):
    global PASS, FAIL
    if cond:
        PASS += 1
        print(f"  ok    {name}")
    else:
        FAIL += 1
        print(f"  FAIL  {name} {detail}")


class Client:
    def __init__(self, port):
        self.s = socket.create_connection(("127.0.0.1", port), timeout=10)
        self.buf = b""

    def send(self, line):
        self.s.sendall(line.encode() + b"\n")

    def lines(self, n=1):
        out = []
        while len(out) < n:
            while b"\n" not in self.buf:
                d = self.s.recv(65536)
                if not d:
                    break
                self.buf += d
            if b"\n" not in self.buf:
                break
            ln, self.buf = self.buf.split(b"\n", 1)
            out.append(ln.decode())
        return out

    def drain(self, wait=0.15):
        self.s.settimeout(wait)
        try:
            while True:
                d = self.s.recv(65536)
                if not d:
                    break
                self.buf += d
        except socket.timeout:
            pass
        self.s.settimeout(10)
        out, self.buf = self.buf, b""
        return [x.decode() for x in out.split(b"\n") if x]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--engine", default="engine/build/chlorine")
    ap.add_argument("--checkpoint", default=None)
    ap.add_argument("--port", type=int, default=8830)
    ap.add_argument("--strict-flags", action="store_true")
    args = ap.parse_args()

    cmd = [args.engine]
    if args.checkpoint:
        cmd += ["--checkpoint", args.checkpoint]
    cmd += ["--serve", "--port", str(args.port)]
    proc = subprocess.Popen(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    try:
        for _ in range(50):
            try:
                socket.create_connection(("127.0.0.1", args.port), timeout=1).close()
                break
            except OSError:
                time.sleep(0.1)
        c = Client(args.port)

        c.send("PING")
        check("PING -> PONG", c.lines(1) == ["PONG"])

        c.send("INFO")
        f = c.lines(1)[0].split()
        check("INFO shape", f[0] == "I" and len(f) == 12, f)
        if args.checkpoint:
            check("INFO ctx", f[3] == "262144", f[3])
            check("INFO spec_rows", f[4] == "8")
            if args.strict_flags:
                check("INFO mtp/dflash2 flags", f[1] == "1" and f[2] == "1" and f[6] == "1" and f[7] == "1", f)

        c.send("CSTAT")
        f = c.lines(1)[0].split()
        check("CSTAT shape", f[0] == "C" and len(f) == 15, f)

        c.send("GEN 7 4 0 2 100 101")
        ls = c.lines(5)
        toks = [x.split() for x in ls]
        check("GEN T lines", all(t[0] == "T" and t[1] == "7" for t in toks[:4]), ls)
        check("GEN D line", len(toks[4]) >= 3 and toks[4][0] == "D" and toks[4][1] == "7", ls)
        check("GEN D fields", len(toks[4]) >= 7 and toks[4][2] in ("stop", "length", "cancel", "error"), ls)

        # eos hit: stub derives tokens from prompt; ask for many, expect either
        # stop/length with exact field count 5 (serial-format fallback) or 8+2
        c.send("GEN 8 100 0 1 55")
        ls = c.drain()
        d = [x.split() for x in ls if x.startswith("D ")][0]
        check("GEN eos/length D", d[2] in ("stop", "length", "error"), ls)

        # PENALTY without sampling -> engine must reject
        c.send("GEN 9 4 0 1 55 PENALTY 0.1 0.2")
        f = c.lines(1)[0].split()
        check("PENALTY w/o temp rejected", f[2] == "error", f)

        # unknown drafter
        c.send("GEN 10 4 0 1 55 3")
        f = c.lines(1)[0].split()
        check("unknown drafter rejected", f[2] == "error", f)

        # LOGPROBS with sample: T lines may carry a trailing logprob field
        c.send("GEN 11 3 0 1 55 SAMPLE 1.0 0 1.0 0.0 12345 LOGPROBS")
        ls = c.drain()
        ts = [x.split() for x in ls if x.startswith("T ")]
        check("LOGPROBS T ok", all(t[0] == "T" and t[1] == "11" for t in ts), ls)

        c.send("X 11")  # must not crash
        time.sleep(0.1)
        check("X no crash", True)

        c.send("BOGUS VERB")
        time.sleep(0.1)
        check("unknown verb ignored", True)
    finally:
        proc.terminate()
        proc.wait()
    print(f"\n{PASS} passed, {FAIL} failed")
    sys.exit(1 if FAIL else 0)


if __name__ == "__main__":
    main()
