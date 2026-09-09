#!/usr/bin/env python3
"""hgn-inspect.py — validate the .hgn container format spec.

Spec (docs/halogen/CHECKPOINT-FORMAT.md, reversed from the halogen 0.1.3
engine binary):

  header (min 0x68 bytes):
    0x00 u32 magic 0x314E4748  ("HGN1" little-endian)
    0x04 u32 version (accepted iff version-1 < 2, i.e. 1 or 2)
    0x08 u64 tensor count
    0x10 u64 tensor-table offset from file start
    0x18 u64 (reserved, not read by the loader)
    0x20 u64 total file size (must equal fstat size)
    0x28 char[64] model name

  tensor entry (stride 0xa0):
    0x00 char[96] name
    0x60 u64 f1 (low u32 = dtype id, high u32 = ?)
    0x68 u64 f2 (dims, probable)
    0x70 u64 f3
    0x78 u64 f4
    0x80 u64 f5
    0x88 u64 data offset
    0x90 u64 data size
    0x98 u64 f8 (high u32 = qparam)

  dtype table (loader .rodata 0x12ea0): 0 bf16, 1 f32, 2 f16, 3 i32, 4 i64,
  5 q4c, 6 fp8r, 7 q8g64, 8 i4l.
"""
import struct
import sys
from collections import Counter

DTYPES = {0: "bf16", 1: "f32", 2: "f16", 3: "i32", 4: "i64",
          5: "q4c", 6: "fp8r", 7: "q8g64", 8: "i4l"}

HDR_ENTRY = 0xA0


def u32(b, o): return struct.unpack_from("<I", b, o)[0]
def u64(b, o): return struct.unpack_from("<Q", b, o)[0]


def main(path, max_entries=None, name_filter=None):
    with open(path, "rb") as f:
        hdr = f.read(0x68)
        if len(hdr) < 0x68:
            sys.exit(f"file too small: {len(hdr)} bytes")
        magic = u32(hdr, 0)
        version = u32(hdr, 4)
        n_tensors = u64(hdr, 8)
        table_off = u64(hdr, 0x10)
        reserved = u64(hdr, 0x18)
        file_size = u64(hdr, 0x20)
        model_name = hdr[0x28:0x68].split(b"\0")[0].decode("utf-8", "replace")

        ok_magic = magic == 0x314E4748
        ok_ver = 1 <= version <= 2
        import os
        real_size = os.fstat(f.fileno()).st_size
        print(f"magic        0x{magic:08X}  {'OK' if ok_magic else 'BAD'}")
        print(f"version      {version}  {'OK' if ok_ver else 'BAD'}")
        print(f"n_tensors    {n_tensors}")
        print(f"table_off    {table_off:#x}")
        print(f"reserved     {reserved:#x}")
        print(f"file_size    {file_size}  (actual {real_size}, {'OK' if file_size == real_size else 'MISMATCH'})")
        print(f"model_name   {model_name!r}")
        if not ok_magic:
            sys.exit(1)

        f.seek(table_off)
        table = f.read(n_tensors * HDR_ENTRY)
        if len(table) < n_tensors * HDR_ENTRY:
            sys.exit(f"table truncated: got {len(table)} of {n_tensors * HDR_ENTRY}")

        dtype_hist = Counter()
        qparam_hist = Counter()
        f1_hi = Counter()
        total_bytes = 0
        bad = []
        max_end = 0
        shown = 0
        print(f"\n{'name':52s} {'dtype':6s} {'q':>6s} {'off':>15s} {'size':>15s}  f2..f5")
        for i in range(n_tensors):
            e = table[i * HDR_ENTRY:(i + 1) * HDR_ENTRY]
            name = e[0:96].split(b"\0")[0].decode("utf-8", "replace")
            f1 = u64(e, 0x60); f2 = u64(e, 0x68); f3 = u64(e, 0x70)
            f4 = u64(e, 0x78); f5 = u64(e, 0x80)
            off = u64(e, 0x88); size = u64(e, 0x90); f8 = u64(e, 0x98)
            dtype = f1 & 0xFFFFFFFF
            qparam = (f8 >> 32) & 0xFFFFFFFF
            dtype_hist[dtype] += 1
            qparam_hist[qparam] += 1
            f1_hi[f1 >> 32] += 1
            total_bytes += size
            if off + size > file_size:
                bad.append((name, off, size))
            max_end = max(max_end, off + size)
            if (max_entries is None and shown < 40) or \
               (max_entries is not None and shown < max_entries) or \
               (name_filter and name_filter in name):
                dims = [f2, f3, f4, f5]
                dims = [d for d in dims if d]
                print(f"{name:52s} {DTYPES.get(dtype, str(dtype)):6s} {qparam:>6d} {off:>15d} {size:>15d}  {dims}")
                shown += 1

        print(f"\ntensors      {n_tensors}")
        print(f"data bytes   {total_bytes} ({total_bytes / 2**30:.2f} GiB)  max_end {max_end:#x}")
        print(f"dtype hist   " + ", ".join(f"{DTYPES.get(d, str(d))}={c}" for d, c in sorted(dtype_hist.items())))
        print(f"qparam hist  " + ", ".join(f"{q}={c}" for q, c in sorted(qparam_hist.items())))
        print(f"f1_hi hist   " + ", ".join(f"{v}={c}" for v, c in sorted(f1_hi.items())[:8]))
        if bad:
            print(f"OUT OF BOUNDS: {len(bad)}")
            for n, o, s in bad[:5]:
                print(f"  {n} off={o} size={s}")


if __name__ == "__main__":
    args = sys.argv[1:]
    max_entries = int(args[1]) if len(args) > 1 else None
    name_filter = args[2] if len(args) > 2 else None
    main(args[0], max_entries, name_filter)
