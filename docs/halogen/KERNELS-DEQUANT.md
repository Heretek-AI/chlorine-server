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

## i4l — row structure known, element order OPEN

- Sizes resolve **exactly** as interleaved rows: `[rows × (cols/2 payload +
  extra)]` (qkv `10240 × (2560+40)` = 26,624,000 ✓; down `5120 × (8704+136)`
  = 45,260,800 ✓) — same row-interleaved family as q4c.
- Per-row extra = cols/128 bytes ⇒ **one scale per 128 columns** (40 or 136
  scales/row). Scale byte encoding presumed e4m3 like q4c `[TO CONFIRM]`.
- Element values: signed int4 two's complement (same code-8-never-occurs
  signature).
- **Element order is NOT row-major**: raw int4 of the i4l twin does NOT
  correlate with the q4c twin's row 0 (corr ≈ 0.001) — the "L" is a GEMM-
  tiled layout consumed by `k_gemm_i4` (prefill W4A4 path) and
  `k_gemv<2,*>`. Cracking it requires the address arithmetic in
  `k_gemm_i4<0>` (`/tmp/opencode/gemm0.asm`, ~2000 lines) `[OPEN]`.
- qparam `0x10100` on every i4l tensor: likely `(flags<<16)|0x100`; exact
  field semantics `[OPEN]`.

## Validation summary

- fp8r: reproduced base-model row0 exactly (std 0.01735 / absmax 0.06885);
  absmax→448 scaling exact.
- q4c: corr 0.987–0.988 vs base model rows 0–39 (int4 quantization error
  only); e4m3 scale bytes match LSQ-implied scales.
- i4l: layout open (see above).
- Base reference: `Qwen/Qwen3.8-27B` shard 1 on evileye at
  `~/Projects/models/qwen-base/`.
