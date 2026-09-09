# halogen 0.1.3 — quantized weight formats (dequant)

Status per format, reversed from `k_dequant<0,1,2>` / `k_gemv` ISA (obj5.so)
plus empirical validation against the real checkpoint. Twin-tensor
correlations and byte accounting; `[OPEN]` = not yet pinned.

## Host dispatch (decomp FUN_005b6370)

- dtype 5 `q4c`: qparam 0 → template A, qparam 1 → template B
- dtype 6 `fp8r`: single template
- dtype 8 `i4l`: NOT handled by k_dequant — consumed directly by the W4A4
  prefill path (`k_gemm_i4`, `k_gemv<2,*>`)

Kernel geometry: block 256 threads, grid.x = elements; both dequant and gemv
first build a 256-entry LDS table, then stream bytes.

## fp8r — CONFIRMED

- Payload: `rows × cols` bytes, row-major, fp8 **e4m3** (s|eeee|mmm, bias 7,
  subnormal m/8·2⁻⁶, no inf, NaN 0x7f/0xff).
- Scales: `rows × 2` bytes **bf16, at the tensor tail**.
- `w[r][c] = fp8e4m3(payload[r·cols+c]) × scale[r]`.
- Quantization convention (verified): per-row absmax scaled so
  `max|fp8val| = 448` — row absmax / scale = 448.0 exactly in the release
  checkpoint.
- Kernel mechanics: 256-entry LDS LUT of e4m3→f32 built arithmetically
  (±(1+m/8)·2^(e−7)); 2 bytes/thread/iter; result RNE-rounded to bf16
  (`x + (x>>16 & 1) + 0x7fff` then take hi16).

## q4c — CONFIRMED (validated vs base model, corr 0.988)

Layout: `[64-byte codebook][row 0: payload+scales][row 1: payload+scales]…`
— payload and scales are **interleaved per row**:

- **Codebook header (64 B)**: 16 f32, per-tensor — this is **NVFP4**:
  `cb = ±e2m1 × 2c` where e2m1 = {0,.5,1,1.5,2,3,4,6}; i.e. cb[0..7] =
  {0,1,2,3,4,6,8,12}·c and cb[8+i] = −cb[i] (sign lives in the nibble).
  `c ≈ 1.8169e-4` for `layers.0.mlp.down_proj`. Nibbles index cb DIRECTLY
  (sign-magnitude; code 8 = −0 exists as a redundant zero encoding — that is
  why its histogram count is 0).
- **Per row** (`[rows, cols]`, row-major): `cols/2` bytes of **sign-magnitude
  e2m1 payload** (element 2k = low nibble, 2k+1 = high nibble; magnitude
  codes 0..7 = e2m1 index, bit 3 = sign), followed by `cols/16` bytes of
  **e4m3 scales** (one per 16-element group).
- `w[r][c] = cb[nib(r,c)] × fp8e4m3(scale[r][c/16])`.
- Validation: dequant of `layers.0.mlp.down_proj` rows 0–39 vs the base
  model (`Qwen/Qwen3-27B` shard 1 bf16) — corr 0.987–0.988 (pure int4
  quantization error), scale bytes decode to the LSQ-implied scales.
- Kernel mechanics match: deq2 builds the e4m3→bf16 LUT, reads payload
  bytes at ±256 offsets (two 16-element groups per thread-iteration) and
  u16 scale pairs.

## i4l — 2026-09-09 deep probe results (element order still OPEN)

- Sizes resolve as `[rows × (cols/2 payload + cols/128 extra)]`; the extra
  block decodes as **all-positive finite fp16** (u16 LE, 68/row for
  `[5120,17408]`) — but the values do NOT match base chunk-absmaxes under
  identity or simple permutations (identity med ratio 0.98, ±45% spread;
  the 98.8% "near-exact set match" is a density artifact).
- **k_actq fully decoded** (obj7.so): per (row, 256-col chunk-pair);
  64 lanes × 4 bf16 loads; group absmax xor-butterfly in f32; u8 codes to a
  per-tensor cached buffer, u16 scales to a shared per-call buffer; padding
  rows zero-filled. Row pointer = base + (nchunkpairs·row + pair)·128 — the
  same chunk-pair structure as the gemm staging.
- **k_gemm_i4 staging**: two 32-row×128 B copies per iteration (acts +
  weights), global row stride = K/2 (natural 16-B records), LDS rows 0x90 B
  (128+16 pad), WMMA iu4 16x16x16, epilogue cvt_f32_i32 + fma_mix with the
  packed act scales (u16→bf16) — integer accumulation, scales at the end.
- **Element order falsified for row 0** vs base W: natural row-major
  (sign 0.502, rankcorr −0.002), W^T reading (0.41), 32×256 tiles at head +
  every 128-B offset in 512 KB (LS rel-residual ≥ 0.9956), sign-magnitude
  reading (0.497), FFT xcorr over the full 89M-nibble stream (no dominant
  peak). The mapping is a GEMM-fragment tiling not derivable from the local
  staging window; needs the LDS-consumption (WMMA fragment read) side of
  gemm0.asm, or a differential probe. Probes: /tmp/opencode/i4l_*.py.
- qparam `0x10100` on every i4l tensor: exact field semantics `[OPEN]`.

## Validation summary

- fp8r: reproduced base-model row0 exactly (std 0.01735 / absmax 0.06885);
  absmax→448 scaling exact.
- q4c: corr 0.987–0.988 vs base model rows 0–39 (int4 quantization error
  only); e4m3 scale bytes match LSQ-implied scales.
- i4l: layout open (see above).
- Base reference: `Qwen/Qwen3.8-27B` shard 1 on evileye at
  `~/Projects/models/qwen-base/`.
