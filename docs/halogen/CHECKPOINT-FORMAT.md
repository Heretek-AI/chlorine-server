# .hgn checkpoint format

Reversed from `checkpoint.cpp` (loader `FUN_005bacf0`) and **validated
against the real checkpoint** `qwen3.8-27b-p1w4d-d2.hgn`
(35,865,565,184 bytes, 1352 tensors, model name `Qwen3.8-27B-p1-d2`,
version 2) — header, table, bounds, dims and name semantics all confirmed.
Inspector: `converter/hgn-inspect.py`.

## File header (0x68 = 104 bytes; table usually starts right after)

| Off | Type | Field | Confirmed value (release ckpt) |
|---|---|---|---|
| 0x00 | u32 | magic `0x314E4748` (bytes `48 47 4E 31` = `"HGN1"`) | ✓ |
| 0x04 | u32 | version (accepted iff `version-1 < 2` ⇒ 1 or 2) | 2 |
| 0x08 | u64 | tensor count | 1352 |
| 0x10 | u64 | tensor-table offset from file start | 0x68 |
| 0x18 | u64 | **first tensor data offset** (data blob start) | 0x34d80 |
| 0x20 | u64 | total file size (must equal fstat size) | 35865565184 |
| 0x28 | char[64] | model name | `Qwen3.8-27B-p1-d2` |

After validation the file is mmap'd and `hipHostRegister`ed (0xa=Mapped,
fallback 0x2=Default) so the GPU reads weights **in place** from pinned host
memory; `hipHostGetDevicePointer` yields the device base for per-tensor
device pointers.

## Tensor table

`n_tensors` entries, stride **0xa0**, at `header[0x10]`:

| Off | Type | Field |
|---|---|---|
| 0x00 | char[96] | name (NUL-terminated, ≤95) |
| 0x60 | u64 | **f1 = (ndim << 32) \| dtype_id** (ndim 1..3 seen; dims live in f2..f5) |
| 0x68 | u64 | dim 0 |
| 0x70 | u64 | dim 1 |
| 0x78 | u64 | dim 2 |
| 0x80 | u64 | dim 3 (unused so far) |
| 0x88 | u64 | data offset in file |
| 0x90 | u64 | data size in bytes (offset+size ≤ file size) |
| 0x98 | u64 | **f8: high u32 = qparam** (65792 = 0x10100 for i4l, 0 otherwise; low u32 unused) |

Dtype id table (9 entries, .rodata 0x12ea0): 0 `bf16`, 1 `f32`, 2 `f16`,
3 `i32`, 4 `i64`, 5 `q4c`, 6 `fp8r`, 7 `q8g64`, 8 `i4l`.

## Storage layout per dtype (from tensor sizes in the release checkpoint)

- **bf16** — 2 B/elem, plain.
- **fp8r** — 1 B/elem fp8 + **20480 B of per-row scales** appended
  (e.g. `[10240,5120]`: 52449280 B = 52428800 data + 20480 = 10240 × 2 B
  bf16 scale per row). Row-major, scale at the tail.
- **q4c** — 4-bit payload + f32 group scales: `[5120,17408]` → 50135104 B ≈
  89,128,960×4 bits + 5,570,624 B ≈ (numel/64 groups)×4 B (group 64),
  Δ64 header. NVFP4-imported values (unsloth/Qwen3.8-27B-NVFP4 per HF note).
- **i4l** — 4-bit layout variant + scales: `[10240,5120]` → 26624000 B =
  52,428,800×4 bits + 409,600 B (40 B per output row of 5120 → 5120/64=80
  groups × 4 B f32 scale per group? exact grouping TBD from k_dequant/i4l
  kernels). qparam 0x10100 encodes group/rot flags ("unsupported I4L group",
  "mixed I4L rot flags" in the binary).
- **q8g64** — 8-bit + group-64 scales (not present in the release ckpt).
- Exact byte layouts to be finalized by disassembling `k_dequant<0,1,2>`,
  `k_gemv`, `k_gemm_i4` (Phase 5).

## Quantization policy in the release checkpoint

Every trunk projection ships **two tensors**: the wide dtype used by decode
streams and a `.weight.i4l` 4-bit twin used by the prefill path (the "W4A4
fenced to prefill" design; env `HALOGEN_W4A4=64`, `HALOGEN_W4A4_EXCL`).

| dtype | GiB | share | where |
|---|---|---|---|
| bf16 | 2.754 | 8.2% | embed_tokens, all norms, small projections, codebooks |
| fp8r | 9.899 | 29.6% | lm_head, linear_attn projections, self_attn projections |
| q4c | 9.244 | 27.7% | all MLP gate/up/down (56/64 layers), drafter+MTP weights |
| i4l | 11.505 | 34.4% | `.i4l` twins of every quantized projection |

The 8 MLP layers stored as fp8r instead of q4c = the `HALOGEN_W4A4_EXCL`
exclusion set.

## Model architecture (from tensor names/dims)

- Trunk: **64 layers** = 48 DeltaNet (`linear_attn`) + 16 full attention
  (`self_attn`); every layer has an MLP (gate/up `[17408,5120]`, down
  `[5120,17408]`). Hidden 5120.
  - DeltaNet: `in_proj_a/b [48,5120]` bf16, `in_proj_qkv [10240,5120]`,
    `in_proj_z [6144,5120]`, `out_proj [5120,6144]` (fp8r+i4l), `A_log
    [48]`, `dt_bias [48]`, `conv1d [10240,1,4]`, `norm [128]`.
  - Attention: `q_proj [12288,5120]` (96 heads×128), `k_proj/v_proj
    [1024,5120]` (8 KV heads), `o_proj [5120,6144]`, `q_norm/k_norm [256]`.
- `embed_tokens [248320,5120]` bf16, `lm_head [248320,5120]` **fp8r**,
  final `norm [5120]`.
- **MTP head** (1 layer): full-attention layer (q 96h) + MLP +
  `mtp.fc [5120,10240]` bf16, `pre_fc_norm_{hidden,embedding} [5120]`,
  `mtp.norm [5120]`.
- **DFlash2 drafter** (5 layers): attention with 32 q heads (`q [4096,5120]`,
  `k/v [1024,5120]`, `o [5120,4096]`), MLP 17408, plus per-layer
  `attention_conv`/`mlp_conv` (`base_kernel [2,2,5120]` bf16 +
  `kernel_projection [1280,5120]` q4c — dynamic conv-kernel generation),
  `fc [5120,25600]` q4c (5-way hidden concat), `hidden_projection [256,5120]`,
  **`predecessor_codebook` / `successor_codebook` [248320,256] bf16**,
  `draft_vocab_map [98304] i32` (draft-vocab subset of 248320),
  `draft_lm_head [98304,5120]` q4c.

### drafter.config (i32[21])

```
[0]  = 1        schema version (engine refuses otherwise)
[1]  = 5        drafter layer count
[2]  = 5120     hidden (must equal trunk)
[3]  = 17408    mlp intermediate
[4]  = 128      head dim
[5]  = 32       q heads (k 1024/128 = 8 kv heads)
[6]  = 8        ?  (verify rows / kSpecRows?)
[7]  = 8        ?  (block/depth cap candidates)
[8]  = 2        block_size? (decomp validated 2..kSpecRows)
[9]  = 16       top-K shortlist (k_dr_topk<16>, k_dr_select<16>)
[10] = 256      codebook width (matches codebooks/hidden_projection)
[11] = 16       sliding window?
[12] = 248070   draft-vocab high-water mark? (vocab 248320)
[13] = 2048     snapshot align
[14] = 0        ?
[15] = 5        tap count (decomp: len == tap_count + 16)
[16..20] = 5, 19, 33, 47, 61   trunk layers tapped (every 14th)
```

`drafter.config_f32` (f32[2]) = `(1e7, 1e-6)`; `drafter.alpha_seed`
(f32[8]) = `(0.0, 0.844, 0.761, 0.769, 0.744, 0.786, 0.768, 0.831)` —
acceptance rates for depths (fallback row in the binary:
0.809/0.746/0.798/0.772/0.770/0.849; `[0]` = 0.0 is outside (0,1) — the
loader validates; exact indexing TBD in the spec pass).

`draft_vocab_map` = identity-on-[0,98304)? min 0, max 248319, first/last
entries ascending — full mapping TBD (98304 of 248320 tokens get fast-path
draft readout).

## Consistency errors (from the loader/validator)

`schema mismatch - engine expects 1; reconvert the checkpoint`,
`tap count inconsistent with the entry size`,
`hidden size does not match the trunk`,
`block_size outside [2, kSpecRows] - the verify batch cannot hold it`,
`sliding_window smaller than the block`,
`q_heads not a multiple of kv_heads`,
`checkpoint: {bad magic, unsupported version, truncated file, entry out of
bounds, no tensor named %s}`.

## Load performance

`checkpoint: registered %.1f GB in %.1f s (%.1f GB/s)` — pinning the mapped
file is the startup cost (~35.9 GB).
