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

## WMMA iu4 fragment maps (empirical, 2026-09-09 probe)

Derived with a raw-LLVM-IR probe kernel (`@llvm.amdgcn.wmma.i32.16x16x16.iu4`
exists; compile `.ll` → `.o` with
`clang -target amdgcn-amd-amdhsa -mcpu=gfx1151 -c`, then **`ld.lld -shared`**
— hipModuleLoad requires a linked shared object, a bare relocatable fails with
hipErrorNoKernelImageForDevice). Probes: /tmp/opencode/wmprobe3.ll,
wmhost.cpp, wmscan.cpp, amap.cpp, joint.cpp + the *.bin dumps.

- **D** (16×16 i32): cell (t′, i) = (m = i + 8·(t′≥16), n = t′&15); threads
  16-31 idle in W32. (Confirmed by both A and B one-hot sweeps — the earlier
  (m = t′&15, n = i+…) reading was wrong.)
- **Shared k-encoding (confirmed by value-multiplication test, 2026-09-09
  wmprobe6/wmhost8 sweep)**: operand position (thread T, vgpr V, nibble N)
  ↔ k = 8·V + N for BOTH A and B; the 4-bit nibble = the element VALUE
  (wmma reads nibble values, not bits). Setting bit J of a vgpr sets
  nibble (J>>2) to value 2^(J&3); two operands multiply (D-cell val =
  A-val × B-val) exactly when their (vgpr, nibble) positions match.
- **A**: element (m, k) → thread T = m&15, vgpr k>>3, nibble k&7 — the
  thread's 2 vgprs = one row's 16 k's ascending. Threads 16-31 idle.
- **B**: element (k, n) → thread T = n&15, vgpr k>>3, nibble k&7 — the
  thread's ENTIRE operand (2 vgprs = 16 k's) = the n-column's 16
  consecutive k's (k = 16h..16h+15 for the h-th wmma k-step). Threads
  16-31 idle. (Thread 0 = the n=0 column, confirmed over all (V, N).)

**Consequence for i4l (REVISED 2026-09-09, later probe)**: the B operand is
ONE weight row's 16 consecutive k's — so the staged/global record (16 B) is
one n-row's k's [16h, +16) in ascending order and **the device weights are
[N, K]-major natural** (row stride K/2 = 8704 ✓ s31). The engine reads the
i4l bytes RAW (the host loader copies them; the per-row 4-B is de-skipped).
The [K, N]-major and n-pair/k-half hypotheses are dead.

**The failed round(W/s) tests are explained: the quantization is
codebook/compensated, not linear.** Evidence:
- magnitude multiset of file codes ≈ clip(round(W/s_file), ±7) but the
  per-position pairing fails everywhere (sign-match 0.41 = random for
  every row/k/row-permutation/k-shift/layout variant tested);
- the scales decode as fp16 with a FIXED exponent 2^-8 (odd bytes
  constant 0x1D) — i.e. 8-bit mantissa × 2^-8, values ≈ absmax/5.3-7.0
  (looser than absmax/7 → outlier clipping);
- file layout = 5120 rows × (4 B + 8704 B) + 5120 × 136 B tail; sizes
  exact (45,260,800 = 5120·8708 + 5120·136);
- the per-row 4-B decodes as 2 fp16 fixed-exp values ≈ scales × 1.11
  (e.g. 0.00559, 0.00580) — same format as the scale block;
- cross-row constraint solving (k must satisfy code[p] = round(W[n,k]/s)
  for ALL n) yields ZERO candidates → the codes are per-row-compensated
  (GPTQ-style) or LUT-mapped, NOT round(W/s).

**Load-time dequant kernels (k_dequant<0/1/2> in obj5.so)** decode SOME
weight dtypes to bf16 at load (the u16-typed weights in k_gemv = PKt):
- `<2>`: payload = raw bytes → 256-entry e4m3 decode LUT built in LDS
  (sign = tid<0x80, exp = (tid>>3)&15, man = tid&7, RNE→bf16), then
  out = LUT[code] × fp16 scale (global_load_d16_b16), RNE→bf16 stores.
  = the fp8r loader.
- `<1>`: same shape but the 256-entry table is LOADED from arg0 (an
  explicit in-file/in-model codebook, fp16 entries converted to bf16) —
  byte codes + explicit codebook + fp16 scales.
- `<0>` (732 B): unexamined; likely the 4-bit variant.
- The forward dispatch switches on a global quant-mode (`DAT_005c4480`,
  trailing-zero count): cases 2-8 = dequant-then-gemm variants, default =
  the gemv path. 8 quant modes total.
- Launchers: FUN_0056b310→k_dequant<0>, FUN_0056b3c0→<1>,
  FUN_0056b490→<2> (grid ((n+3)/4, 40) — 40 = N/128 row-tiles).

**Next step**: read k_dequant<0>'s 732 B — if it indexes a 16-entry LUT
(arg0) that is one of the file sections, the i4l = LUT-int4 and the
codebook location + the exact scale addressing fall out of its addressing
math directly.

**Additional layout probes (2026-09-09, late)**: stride-8712 variant
(8704 payload + 8 B in-row + 128-B/row tail block of 64 fp16 all-positive
≈-scales) — the in-row 8 B is NOT fp16 scales (junk: 670, -9744, …);
tail-64-fp16 sign-match also 0.41, and absmax/7-vs-scale ratios scatter
0.78-1.31 — **the scales are for PERMUTED k-chunks: the file k-order =
GPTQ act-order (data-dependent permutation), NOT the HF natural order.**
The engine is self-consistent (its acts, actq, and gemm all use the file
order), which is why it validates.

**Recovery plan for the permutation P**: superseded — SOLVED 2026-09-09
(late session). The "permutation" is not an act-order permutation at all:

## I4L SOLVED: Hadamard-rotated int4 (QuaRot-style)

The i4l tensor is the SAME weights quantized after an orthogonal 256-block
Hadamard rotation. Element-wise it is uncorrelated with W (corr ≈ 0.001
vs W, ≈ 0.002 vs q4c codes) but per-row energy is preserved to the
quantization error (ratio 1.015-1.019), and blockwise reconstruction with
the normalized Hadamard-256 gives corr(W_rec, W) = 0.9920-0.9924 across
rows. Engine correctness follows: acts are rotated by the same H (the
k_actq butterfly IS the fast Hadamard transform), so W'·a' = W·a exactly.

**File layout (i4l, universal — verified on down_proj AND gate_proj)**:
- [N rows × K/2 bytes of 4-bit two's-complement codes (lo nib = even k')]
  ++ [separate scale block at the end: N × (K/256) fp16 LE]. Tensor size
  = N·K/2 + N·(K/256)·2 exactly (45,260,800 for BOTH [5120,17408] =
  5120·8704 + 5120·136 and [17408,5120] = 17408·2560 + 17408·40).
  (An earlier note claimed per-row interleaved scales — wrong; the block
  is contiguous after all payload rows.)
- scales = one fp16 per (n, 256-k' chunk): 8-bit mantissa × tensor-shared
  fixed exponent (2^-8 for layer 0 down/gate; bytes = [b, 0x1D] LE, fp16
  value = (1 + (256+b)/1024)·2^-8); scale = absmax(rotated chunk)/7
  EXACTLY (absmax/s = 7.000 at p1/p50/p99 on both tensors).
- dequant: W'[n,k'] = code[n,k'] · s[n][k'>>8]
- reconstruction: W[n, 256c:(c+1)·256] = W'[n, same] @ Hadamard256/16
- gate_proj verification: corr 0.9922-0.9931 (rows 0-3), absmax/s = 7.000.

**Shadow tensors**: every i4l tensor has a same-dims q4c (dt=5) twin
(e.g. layers.0.mlp.down_proj.weight dt=5 + .weight.i4l dt=8). q4c =
UNROTATED linear int4 (corr 0.987, decode/gemv path); i4l = rotated copy
(W4A4/prefill gemm path). Both loaded → the ~18 GB weight pool.

**Hadamard details**: H256 = the standard unnormalized Hadamard (Sylvester)
divided by 16; H symmetric so left/right multiply agree. corr peaks at
shift 0 (shift-1 corr = 0.001), absmax/s = 7.000 post-hoc confirms both
the chunking and the scale semantics. Residual 0.992 (not 1.0) = the
4-bit quantization error only (q4c shows the same 0.987).

**Probes**: /tmp/opencode/i4l_{kn,npair,permsweep,final*,split*,score,
tileperm,rot,had,verify,entry,names}.py; kmap2.txt (full WMMA B-map sweep);
q4c_dump.py; the dequant/gemv kernel decodes (obj5.so via
/var/cache/lemonade/.../llvm-objdump).

## Validation summary

- fp8r: reproduced base-model row0 exactly (std 0.01735 / absmax 0.06885);
  absmax→448 scaling exact.
- q4c: corr 0.987–0.988 vs base model rows 0–39 (int4 quantization error
  only); e4m3 scale bytes match LSQ-implied scales.
- i4l: SOLVED — Hadamard-rotated int4, reconstruction corr 0.9920-0.9924
  (rows 0-7, down_proj), scale = absmax(rot-chunk)/7 exact, layout above.
- Base reference: `Qwen/Qwen3.8-27B` shard 1 on evileye at
  `~/Projects/models/qwen-base/`.
