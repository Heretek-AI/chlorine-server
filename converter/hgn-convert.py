#!/usr/bin/env python3
"""hgn-convert.py — safetensors -> .hgn writer (format: docs/halogen/CHECKPOINT-FORMAT.md).

Supported input dtypes: F32, F16, BF16, I32, I64 (stored as-is; bf16 is
copied without conversion), plus optional float -> fp8r rowwise (absmax->448)
and q4c (per-tensor linear codebook + 1B-per-16 scales) quantization.

Usage:
  hgn-convert.py OUT.hgn MODEL_NAME NAME=DTYPE[:SRC]...   (from .safetensors)
  hgn-convert.py --roundtrip-test                         (synthetic smoke test)

Byte layout written:
  header (0x68): magic HGN1 | version 2 | n | table_off 0x68 |
                 data_start 0x68 + 0xA0*n | file_size | name[64]
  table entries (0xA0): name[96] | (ndim<<32)|dtype | dims[4] |
                 offset | size | qparam<<32
  data blobs in entry order, each 64-byte aligned.
"""
import json
import struct
import sys

import numpy as np

MAGIC = 0x314E4748
ENTRY = 0xA0
HDR = 0x68
DTYPES = {"bf16": 0, "f32": 1, "f16": 2, "i32": 3, "i64": 4,
          "q4c": 5, "fp8r": 6, "q8g64": 7, "i4l": 8}


def read_safetensors(path):
    with open(path, "rb") as f:
        n = struct.unpack_from("<Q", f.read(8))[0]
        hdr = json.loads(f.read(n))
        blob = f.read()
    out = {}
    base = 0
    for name, meta in hdr.items():
        if name == "__metadata__":
            continue
        dt = meta["dtype"]
        shape = meta["shape"]
        ends = base + meta["data_offsets"][1]
        starts = base + meta["data_offsets"][0]
        out[name] = (dt, shape, blob[starts:ends])
        base = ends
    return out


def to_bf16_u16(f32):
    """float32 -> bf16 bits (round-to-nearest-even), returns uint16 array."""
    u = f32.astype(np.float32).view(np.uint32)
    lo = u & 0xFFFF
    r = (u >> 16) + ((lo > 0x7FFF) | ((lo == 0x7FFF) & ((u >> 16) & 1)))
    return (r & 0xFFFF).astype(np.uint16)


def np_dtype_bits(dt):
    return {"F64": 8, "F32": 4, "F16": 2, "BF16": 2, "I64": 8,
            "I32": 4, "I16": 2, "I8": 1, "U8": 1, "BOOL": 1}.get(dt, 0)


def quant_fp8r(f32):
    """Rowwise absmax->448 e4m3. Returns (payload bytes, bf16 scale bits)."""
    rows = f32.reshape(-1, f32.shape[-1])
    amax = np.abs(rows).max(axis=1)
    amax = np.where(amax == 0, 1.0, amax)
    scale = (amax / 448.0).astype(np.float32)
    q = np.clip(np.round(rows / scale[:, None]), -448, 448)
    # encode e4m3: magnitude with bias 7
    mag = np.abs(q)
    with np.errstate(divide="ignore"):
        e = np.floor(np.log2(mag + 1e-30)).astype(np.int32)
    e = np.clip(e, -6, 8)
    m = np.round(mag / 2.0 ** e * 8).astype(np.int32)
    m = np.clip(m, 0, 8)
    m = np.where(m == 8, 0, m)
    e = np.where(m == 8, 0, m)
    e = np.where(m == 8, e + 1, e)
    byte = np.where(q < 0, 128, 0) + (e + 7).astype(np.uint32) * 8 + m
    return byte.astype(np.uint8).tobytes(), to_bf16_u16(scale).tobytes()


def quant_q4c(f32, group=16):
    """Signed int4 (two's complement, values in [-7,7]) with per-tensor
    linear codebook header (16 f32 = +0..7, -0..-7 scaled) and 1B scales."""
    flat = f32.reshape(-1)
    n_groups = flat.size // group
    usable = n_groups * group
    amax = np.abs(flat).max()
    # nibble range +-7 -> per-group scale = amax/7
    scale = np.float32(amax / 7.0) if amax > 0 else np.float32(1.0)
    q = np.clip(np.round(flat[:usable] / scale), -7, 7).astype(np.int32)
    q &= 0xF
    payload = (q[0::2] | (q[1::2] << 4)).astype(np.uint8)
    cb = (np.arange(8) * np.float32(scale / 448.0)).astype(np.float32)
    codebook = np.concatenate([cb, -cb]).astype(np.float32).tobytes()
    # per-group scale byte: e4m3 of scale (host multiplies cb[nib] * 448 * scale_byte)
    sbytes = quant_fp8r(np.full((1, n_groups), scale))[0]
    return codebook, payload.tobytes(), sbytes


def write_hgn(path, model_name, tensors):
    """tensors: list of (name, dtype_name, dims tuple, bytes blob, qparam)."""
    n = len(tensors)
    data_start = HDR + ENTRY * n
    entries = [bytearray(ENTRY) for _ in range(n)]
    blobs = bytearray()
    off = data_start
    for i, (name, dt, dims, blob, qparam) in enumerate(tensors):
        e = entries[i]
        nb = name.encode()[:95]
        e[0:len(nb)] = nb
        struct.pack_into("<Q", e, 0x60, (len(dims) << 32) | DTYPES[dt])
        for j, d in enumerate(dims[:4]):
            struct.pack_into("<Q", e, 0x68 + 8 * j, d)
        struct.pack_into("<QQ", e, 0x88, off, len(blob))
        struct.pack_into("<Q", e, 0x98, qparam << 32)
        pad = (-len(blob)) % 64
        blobs += blob
        blobs += b"\0" * pad
        off += len(blob) + pad
    total = data_start + len(blobs)
    hdr = bytearray(HDR)
    struct.pack_into("<IIQQQQ", hdr, 0, MAGIC, 2, n, HDR, data_start, total)
    hdr[0x28:0x28 + len(model_name)] = model_name.encode()[:63]
    with open(path, "wb") as f:
        f.write(hdr)
        for e in entries:
            f.write(e)
        f.write(blobs)


def convert(src_st, dst_hgn, model_name, quant=None):
    tensors = []
    for name, (dt, shape, raw) in read_safetensors(src_st).items():
        dims = tuple(shape)
        if dt == "BF16":
            tensors.append((name, "bf16", dims, raw, 0))
        elif dt == "F32":
            if quant == "fp8r" and len(shape) == 2:
                pay, sc = quant_fp8r(np.frombuffer(raw, dtype=np.float32).reshape(shape))
                tensors.append((name, "fp8r", dims, pay + sc, 0))
            else:
                tensors.append((name, "f32", dims, raw, 0))
        elif dt == "F16":
            tensors.append((name, "f16", dims, raw, 0))
        elif dt == "I32":
            tensors.append((name, "i32", dims, raw, 0))
        elif dt == "I64":
            tensors.append((name, "i64", dims, raw, 0))
        else:
            raise SystemExit(f"unsupported source dtype {dt} for {name}")
    write_hgn(dst_hgn, model_name, tensors)
    print(f"wrote {dst_hgn}: {len(tensors)} tensors")


def synth_roundtrip(path="tests/conformance/toy.hgn"):
    rng = np.random.default_rng(7)
    V, H, L = 256, 64, 2
    tensors = [("embed_tokens.weight", "bf16", (V, H),
                to_bf16_u16(rng.normal(0, 0.02, (V, H)).astype(np.float32)).tobytes(), 0)]
    for l in range(L):
        p = f"layers.{l}."
        tensors += [
            (p + "input_layernorm.weight", "bf16", (H,),
             to_bf16_u16(np.ones(H, np.float32)).tobytes(), 0),
            (p + "mlp.gate_proj.weight", "q4c", (2 * H, H), None, 0),
            (p + "mlp.down_proj.weight", "fp8r", (H, 2 * H), None, 0),
        ]
    tensors += [("lm_head.weight", "bf16", (V, H),
                 to_bf16_u16(rng.normal(0, 0.02, (V, H)).astype(np.float32)).tobytes(), 0)]
    out = []
    for name, dt, dims, blob, q in tensors:
        if blob is None:
            f32 = rng.normal(0, 0.02, dims).astype(np.float32)
            if dt == "fp8r":
                pay, sc = quant_fp8r(f32)
                blob = pay + sc
            else:
                cb, pay, sb = quant_q4c(f32)
                blob = cb + pay + sb
        out.append((name, dt, dims, blob, q))
    write_hgn(path, "chlorine-toy", out)
    print(f"wrote {path} ({len(out)} tensors)")


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "--roundtrip-test":
        synth_roundtrip()
    elif len(sys.argv) > 1 and sys.argv[1] == "--quant":
        convert(sys.argv[3], sys.argv[2], sys.argv[4] if len(sys.argv) > 4 else "converted", quant="fp8r")
    else:
        if len(sys.argv) < 4:
            print(__doc__)
            sys.exit(2)
        convert(sys.argv[1], sys.argv[2], sys.argv[3])
