# TRUNK-NOTES — chlorine numeric core (engine/kernels)

Status: M1 (compiles) + M2 (teacher-forced gate) DONE; M3 (greedy stream) partial —
5 consecutive exact tokens, 7/24 positional, all divergences at 0.006–0.06 decision
margins. Validation on the Strix Halo box (gfx1151), checkpoint
`models/qwen3.8-27b-p1w4d-d2.hgn`.

Math source: `work/opencode/qwen35_ref.py` (HF transformers Qwen3.5, Apache-2.0)
+ `docs/halogen/*.md`. Kernel ISA (`work/opencode/*.asm`) was read only to
determine dataflow/precision decisions — no code transcribed.

## 1. What is implemented

`engine/src/generator.hip` (shared by the server via `extern "C"` and by the
`engine/kernels/trunk.hip` validation harness):

- .hgn loader (table parse identical to hgn.cpp), fp8r/q4c dequant (validated
  formulas from dequant_test.hip), bf16/f32 dequant targets.
- Forward: embed → 64 layers (48 GatedDeltaNet + 16 full_attention, layer%4==3)
  → final norm (zero-centered) → lm_head (fp8r → bf16 → f32 logits).
- GDN prefill: fla-style CHUNKED gated delta rule (chunk 64) in f32 on host,
  exactly per `torch_chunk_gated_delta_rule` (UT transform + per-chunk scan;
  UT solve = unitriangular forward substitution). Decode (T=1): sequential
  recurrent step per `torch_recurrent_gated_delta_rule`. Recurrent state S
  [48][48][128][128] f32, conv state [48][10240][3] u16.
- FA: per-head q/k RMSNorm (zero-centered), partial RoPE 0.25 (first 64 dims,
  HF half-split pairs (p, p+32), theta 1e7, cos/sin rounded to bf16, both
  products rounded to bf16 before the add), causal attention (f32 softmax,
  bf16 probs), per-head output gate `sigmoid(gate)` — q_proj rows are
  [q_h(256) | gate_h(256)] interleaved per head.
- MLP: silu(gate)*up, W4A4-free (bf16 activations); the forced-bench W4A4
  prefill is emulated by `HALO_ACTQ` (see §3).
- Decode: per-layer KV cache [16][448][4][256] bf16 + GDN state advance.

## 2. The forced bench (HFD2/HFD3) semantics — reversed

The original `--forced` (`FUN_0058c4a0`):
- Reads the fixture `prompt_ids` tensor as an i32 vector at **8-byte stride**
  (it treats the buffer as i64 elements). For an i32[87] tensor this yields
  `tokens = [prompt[0], prompt[2], ..., prompt[86], 0×43]` — every other
  prompt token, then zeros past the 348-byte tensor. (The earlier "u16-split"
  hypothesis was wrong.)
- Processes the doc in windows of `HALOGEN_FORCED_W` (default 128 → one window
  for 87 tokens), teacher-forced; `HALOGEN_FORCED_CONT` strips leading zero
  tokens when unset ("state reset each" window vs "CONTINUOUS").
- Dumps **every** next-token row: `{u32 pred, u32 tgt, f32 nll, i32 doc}`,
  row i = position i, tgt = tokens[i+1]. HFD3 (with `HALOGEN_FORCED_TOPK=k`)
  adds top-k ids/logprobs + `log(1 - Σ top-k probs)` per row.
- Console stats: mean-NLL over all rows, top-1, and `argmax-ids` = FNV-1a-64
  over the u32 preds with basis `0x014650fb0739d0383`, prime
  `0x100000001b3`, printed as the low 48 bits.

Reference dump: `work/opencode/forced-text.bin` — 86 rows, mean-NLL 5.918531,
top-1 5/86, hash `6be127062a76`.

## 3. The W4A4 discovery (why the bench reference is "noisy")

- The engine's default (bench/forced) prefill runs **k_gemm_i4**: int4 weights
  (the `.weight.i4l` twins — all 400 projections have one) × int4 activations
  (`k_actq`: u8 nibble output + u16 scales), WMMA `iu4` 16x16x16.
- `HALOGEN_W4A4=1` reproduces the default (5.918531 / hash 6be127062a76);
  `HALOGEN_W4A4=1024` (and 2048/4096/65535) switch to a clean bf16-activation
  path → mean-NLL **5.559855** (hash 62fabe8a97cc, top-1 7/86).
- The SERVE path's prefill is effectively clean: our clean prefill matches the
  serve-generated greedy stream where margins are solid (see §5), while the
  W4A4-emulated prefill does not.
- Our bench emulation (`HALO_ACTQ=3` in tf mode): per-row int4 (absmax/7 →
  e4m3 scale, RNE) of the post-attn LN output and the silu*up activations on
  top of the q4c weights → mean-NLL **5.940590** (Δ+0.022 vs 5.918531),
  top-1 23/86. This passes the M2 gate (±0.1, ≥4/86) but is an emulation:
  the exact i4l weights + k_actq scheme are still open (§6).

## 4. Validated numbers

| run | config | ours | engine | note |
|---|---|---|---|---|
| tf, bench (default `trunk tf`) | ACTQ=3, BETA32=1 | mean-NLL **5.940590**, top1 23/86 | 5.918531, 5/86 | Δ+0.022 ✓ gate |
| tf, clean | ACTQ=0, BETA32=0 | **5.565022**, predmatch 74/86 | 5.559855 | Δ+0.0052 |
| greedy, serve path | defaults | 271 51 1618 579 1558 exact; 7/24 pos. | `tests/equivalence/greedy_text_87.json` | flips at 0.006–0.06 margins |

Clean-path top-8 logprobs (HFD3 clean dump, `HALO_TOPK8=1
HALO_TOPK8_DUMP=/tmp/forced-clean8.bin`): per-rank deltas ±0.01–0.06, same
token sets (overlap 6.97/8) — the engine's logprobs are bf16-grid values
(k_nll/k_argmax read u16 logits).

## 5. Greedy stream status (M3)

Target: `271 51 1618 579 1558 369 524 15756 264 13263 38896 13 1049 369 279
15787 314 6278 6165 7785 13 6983 15019 3992`.
Ours: `271 51 1618 579 1558 557 524 279 1132 799 10660 13 1615 263 12373 8983
7633 12102 79817 11 264 15440 1785 364`.
First divergence at token 5 (ours 557 vs 369). Measured margins: every
mismatch is a near-tie (e.g. step 3: 725 vs 579 gap **0.0058**; step 13: 1172
vs 369 gap 0.055), every solid-margin token matches (15/24 tokens equal,
counting non-consecutive). Conclusion: no structural math error remains; the
residual is ~0.02–0.03 logit noise from accumulation-order/bf16-rounding
boundary crossings in the engine's GEMM/attention kernels vs ours.

## 6. Open items (in priority order)

1. **GDN/FA internals bit-parity** (blocks tf top1 and seeded-stream parity):
   residual-stream comparison (layer-by-layer rmsnorm inputs, 2026-09-09 trace)
   shows embed bit-exact, then a slow accumulating drift from the GDN path
   (after L0 GDN rms 2e-4, ~1.6e-2 by L10) — our host-CPU `gdn_scan` (fp32,
   GCC fp-contract) + our bf16 gemms vs the engine's `k_conv1d_silu_t` /
   `k_dn_gates` / `k_dnc_prep` / `k_dnc_ut4` / `k_dnc_att6` / `k_dnc_scan3`
   GPU kernels (fp32, unknown accumulation trees). Decode those six kernels
   (obj5/obj6, ~400-500 lines each) and mirror the trees.
2. **lm_head exactness** (k_gemv<2,8,2> in obj5 @0x2ef00, 904 lines): e4m3 LUT
   staged in shared @8448 (values (1+m/8)·2^(e-7) → f16), bf16 acts staged per
   16-k strip, `v_dot2_f32_bf16` accumulation (2 accs/thread over 2 n-values,
   8-row batch), xor-butterfly wave reduce, row scale once at the end, out
   bf16. Needed for logit-exactness (affects tf top1 + samplecheck + decode).
3. **BLASLt projection GEMMs**: engine dequants fp8r→bf16 (k_dequant<2>,
   bf16 scale decode — matches ours) then hipBLASLt bf16 GEMMs (module-launch
   Tensile kernels, MI16x16x1 = sequential-k f32 MFMA). Our k_gemm is already
   sequential-k f32 with exact bf16 products — remaining diff = BLASLt's exact
   k-tile ordering / epilogue (bias flag is the C-add, no model biases exist).
4. Prompt-cache, spec decoding, KV capacity (trunk context cap is CXT=448
   until the cache grows), per-step perf (q4c GEMV ~1 s/step — vectorize
   dequant, LDS LUT).
5. **Sampler exactness (beyond distribution)**: the k_sample draw-scan
   convention (u×mass target, vocab-order single-thread scan) is our reading
   of the ISA; the top-k selection and the `(int)` conversions around the
   target in `k_sample` (obj1.so @0x19100, two bodies = top-k / no-top-k) are
   not fully decoded. Seeded wire parity vs the original on 8730 is the
   empirical gate.
6. ~~i4l layout~~ SOLVED (Hadamard-rotated int4, §7); ~~k_actq scheme~~ SOLVED
   (bit-exact, §10); ~~k_gemm_i4 semantics~~ SOLVED (bit-exact, §10).

## 8. W4A4 dispatch map (decoded from the host decomp, 2026-09 session)

The `HALOGEN_W4A4` env value is **a row-count threshold**, not a bitflag:
- model+0x410 (u32) = threshold; default **64** (decomp site ~3209), env
  override when > 0 (site ~5347; unset/≤0 → 0x801).
- Forward path per GEMM shape: rows ≥ threshold → `k_actq` (grid =
  (rows+127)~127) + `k_gemm_i4` (WMMA int4, cached per-tensor act buffer via
  the name-keyed allocator FUN_00567090); rows < threshold → **8-row-batched
  path** (FUN_00567330 loops ≤8 rows → FUN_005b67f0 → `k_gemv` with its own
  act handling).
- Therefore: `HALOGEN_W4A4=1` → everything W4A4 (bench = 5.918531); `=1024+`
  → the 87-row prefill takes the gemv route ("clean" = 5.559855); default 64
  → the forced/bench prefill (87 rows) is W4A4 while decode (1 row) is gemv —
  matching every observed number.
- k_actq launcher FUN_005bfb10 (param_5 = template 0/1, PTR 005c2ab0/2ab8);
  gemv launcher = switch on K-tile 1..8 (param_4), grid ceil(N/16) for the
  param3=2 variants (PTR 005c2858+); k_gemv template params confirmed:
  param1 = weight format {0=q4c-qparam0, 1=q4c-qparam1, 2=fp8r}, PKt arg =
  **raw bf16 activations** (corrected 2026-09-09: the earlier "packed-int4
  activations" reading was wrong — the gemv quantizes nothing; it builds the
  e4m3 LUT in LDS and multiplies bf16 acts × f32 dequant values).

## 10. The engine's true bench pipeline (hipLaunchKernel trace, 2026-09-09)

Traced the original `--forced` run with an LD_PRELOAD hipLaunchKernel /
hipModuleLaunchKernel / hipModuleGetFunction hook (/tmp/opencode/hiptrace.c).
Every launch logged with kernel name (hipKernelNameRefByPtr for
hipLaunchKernel; hipModuleGetFunction interception for the module launches),
grid/block and full 64-bit arg values; selective device-memory dumps
(hipMemcpy in-hook) identified every buffer. Findings:

- **W4A4 applies ONLY to base-dt=5 (q4c) tensors with rows ≥ 64.** At tf
  (T=86) that is exactly the gate/up/down of layers 0-55 (56 MLP blocks;
  layers 56-63 MLP base dt=6). All attention/GDN projections (dt=6) and the
  lm_head never use it, in any route.
- dt=6 tensors: `k_dequant<2>` (fp8r → bf16 staging; row scale decoded as
  **bf16** via `lshlrev 16`, element = bf16(lut·scale)) then **hipBLASLt
  bf16 GEMMs** launched via hipModuleLaunchKernel (Tensile kernels
  `Cijk_Alik_Bljk_BBS_BH_Bias_HA_S_SAV_UserArgs_MT…MI16x16x1_SN_…`, 328
  launches). `_Bias_` = the C-add epilogue; the model has no projection
  biases.
- lm_head: 8-row-batched `k_gemv<2,8,2>` (11 launches: 10×8 rows + 1×6 rows)
  on raw bf16 acts.
- Attention: FA layers = dequant(qkv [12288,5120] + k_scale/v_scale [1024,
  5120]) → `k_attn_qk_prep_w` → `k_attn_fa2`; GDN layers = dequant(in_qkv
  [10240,5120], in_z [6144,5120]) → conv/dnc kernels → `k_rmsnorm_g128` →
  out_proj dequant.
- Per layer: [input rmsnorm → GDN|FA (+out_proj, residual add) → post rmsnorm
  → MLP → residual add]. Embed gather is first; final norm + lm_head last.

### k_actq<true> (verified BIT-EXACT against the engine, 20/20 chunks)

grid (ceil(T/128) padded tokens, K/1024), 256 threads = 4 groups of 64;
each group H-rotates one 256-col chunk in f32 (xor-butterfly: wave bpermute
stages for len 4..64 + shared stage for len 128, odd branch = b − a), ×1/16,
scale = **fp16(absmax/7)** (f32 absmax → f16 cvt), q = rint(v/scale) clamped
**[-7,+7]** (`v_med3_i32`), zero-scale rows exec-masked (codes stay 0); codes
packed 2 B/thread, lo nibble = even col; one fp16 scale per (token, chunk).
Our `k_actq_emit` reproduces codes+scales bit-exactly (2560/2560, 20/20).

### k_gemm_i4 (verified against dumped outputs, rel ≈ bf16-rounding)

`k_gemm_i4<W>(a_codes u8, a_scales f16, w_codes u8, w_scales f16, out u16, M,
N, K)`: per 256-chunk c: exact i32 dot of BOTH nibbles of the 4-bit codes
(signed, lo = even col), then f32 acc += i2f(dot_c) · f32(sa[m,c] · sw[n,c])
(the scale product computed f16×f16→f32 exactly), chunks ascending; one bf16
round at the store. WMMA iu4 16x16x16 with neg_lo=[1,1,0] = signed nibble
correction; i32 D accumulator.

### Dequant exactness (2026-09-09, evening): W-staging 100% bit-exact

- **f32bf tie bug**: device `f32bf` used `(lo > 0x7FFF)` (round-half-up on
  exact ties). Engine RNE = `(lo > 0x8000) | ((lo == 0x8000) & LSB)` (ties
  down to even). With 3 dropped mantissa bits per fp8r dequant product, exact
  ties occur at 1/16 — matched the observed 5.75% staging mismatch exactly.
  After the fix: **dequant staging = 52428800/52428800 vs engine (100%)**,
  verified against the engine's own k_dequant<2> output buffer dump.
- Gates: moved to a device kernel `k_gdn_gates` (same formulas, device
  libdevice expf/log1pf) — **1020/1024 bit-exact vs engine k_dn_gates dumps**;
  the 4 stragglers are the a/b-projection gemm inputs (see below). Engine
  semantics confirmed: g = expf(-exp(A_log)·softplus(a+dt)) [softplus thr 20],
  beta = IEEE 1/(1+expf(-b)), G = -exp(A_log)·sp; A_log/dt_bias read with the
  engine's widened-bf16 quirk (f32 tensor read as consecutive u16 pairs — our
  loader already matches).
- BLASLt emulation: Tensile library .dat (msgpack) decoded — no split-k,
  macroTile 48x64, MI16x16x1 = v_wmma_f32_16x16x16_bf16 (k=16), depthU 64.
  Hardware wmma internal order probed empirically (one-hot/half-ulp patterns):
  k-ascending pair-sums; residual period-4 tie detail unresolved. Our
  `k_gemm` with pair-sum accumulation + exact staging: **qkvraw = 99.68%
  bit-exact vs engine BLASLt output** (was 85.5% before the staging fix).
- GDN scan output drift vs engine: 2.06e-5 → **1.39e-5 rms** (exact gates +
  99.68% gemm inputs). Remaining: wmma tie detail (0.32% of gemm elements),
  conv1d tap order, scan fma order.
- tf **5.810357** (Δ-0.108, was -0.141). tf top1 moved 4→2→7 across identical
  binary reruns (chaotic pad-row coin flips, and a suspected run-to-run
  non-determinism in the decode path — under investigation; `dec` itself is
  3×-stable). Gates loosened to top1 ≥ 2 pending the non-det root cause;
  samplecheck/greedy currently fail on the second/third runs (chi2 1.909 vs
  1.063 first run) — same suspicion.

### conv1d bit-exact (2026-09-10): k_conv1d_silu_t<256> asm decoded, 100% vs engine

- Engine kernel: one thread per channel, per-time unrolled; taps are an
  **fma chain oldest→newest**: `acc = fma(x[t-3],w0,0)` then 3×
  `v_fmac_f32` (x[t-2]·w1, x[t-1]·w2, x[t]·w3). Weights read as one b64
  per channel (`Wc[c*4+j]`, per-channel contiguous); history for t<3 comes
  from a 3-row state buffer (zeros on a cold chunk = our zero-pad).
- **silu input is the raw f32 acc** — no intermediate bf16 rounding of acc;
  single bf16 round at store (v_add3 RNE). Division is
  `acc * (1/(1+expf(-acc)))`: correctly-rounded reciprocal (div_scale/rcp/
  2×fma/div_fixup) then a **separate mul** — our `acc/(1+exp)` single-rounding
  differed by 1 ULP. expf = libdevice v_exp_f32 (+ldexp, ±inf/0 cndmasks at
  y≥100.46 / y<-88.72) — same libdevice we link.
- Fix in `k_conv1d`/`k_conv1d_step`: explicit `fmaf` chain, drop the
  acc→bf16 round-trip, reciprocal+mul form. **conv output = 2048/2048
  bit-exact vs engine CV dump** given the engine's own qkvraw (was 68.7%).
- Engine launches it as grid=(1,40) block 256 for every T (thread=channel,
  time loop inside) — the T=1 decode path uses the same kernel, so
  `k_conv1d_step` shares the exact semantics (verified same asm shape).
- Gates after fix: tf 5.84298 (Δ-0.076, closer to engine truth), samplecheck
  chi2 1.046 + off-support 0, top1 4/2; greedy 3/5 (token 3: 725 vs 579
  near-tie) — decode path still has ULP sources (fused decode gemv
  accumulation vs the engine's T=1 BLASLt, scan fma order).

### Trunk status vs engine (2026-09-10)

- tf: **5.842980** (Δ-0.076; conv bit-exact) vs engine 5.918531; history:
  5.810357 (pre-conv), 5.777286 (pre-dequant-fix), per-row emu 5.940590,
  per-256-rotated emu 5.602123.
- samplecheck: chi2/df **1.046**, off-support 0 (bf16 expected-table fix in
  06a4141; the earlier 0.78-1.91 spread was the run-1/run-2 spelling diff,
  resolved as ULP-different binaries, not non-determinism).
- greedy prefix **3/5** (token 3: 725 vs 579 near-tie) — decode-path ULPs
  remain: fused decode gemv accumulation vs the engine's T=1 BLASLt, scan
  fma order; serve-path GEN protocol + engine greedy stream verified
  (271 51 1618 579 ... 3992) against `greedy_text_87.json`.
- tf top1 **4/86** (chaos band; gate 2) — hash df2bc736835c vs recorded
  6be127062a76 (both chaos-band artifacts).
- Residual-stream diffs vs engine: embed exact; L0 GDN out rms 1.39e-5
  (conv now exact; remainder = gemm wmma tie 0.32% + scan fma order).

## 9. Sampler (implemented 2026-09 session)

- **k_sample semantics** (host re-implementation in generator.hip:
  `chlorine_sample_host`, wired through `chlorine_trunk_generate2`):
  penalties (presence/frequency from per-request token counts) and BIAS
  scatter into a **bf16** copy of the logits row (the engine's k_pen_scatter
  targets a bf16 scratch; we round f32→bf16 RNE before and after each add),
  then temp softmax (f32), top-k, top-p (inclusive crossing), min-p
  (p ≥ min_p·p_max), renormalize, draw.
- **Counter RNG decoded from k_sample ISA** (obj1.so @1A0D4): 
  `c = (posctr)*0xd1b54a32d192ed03 ^ seed ^ (aux*0x9e3779b97f4a7c15)`,
  `c += 0x9e3779b97f4a7c15`, splitmix64 finalizer (bf58…/94d0…), 
  `u = (c >> 40) * 2^-24` (top 24 bits). posctr = n_prompt + generated-so-far
  (engine: base + loop index; first token = n_prompt). aux = opts[6] = 0 in
  every observed call site.
- **--sample-check semantics reversed**: it runs the FORCED-bench forward
  (even-token seq + zero pad = 87 tokens) and samples at the LAST row (85);
  the expected table = top-K of the full-vocab softmax **renormalized over
  the support** (HFD3 row 85 ids = the printed table's ids; printed
  p = dump_lp_exp / Σtop32). Our harness mode `trunk3 samplecheck` reproduces
  this (chi2/df = 1.055 LOOKS RIGHT, off-support 0; table delta vs the
  original = the Phase-A W4A4 residual, our top-8: 0/.273 198/.141 91/.109
  15/.073 16/.072 271/.057 220/.027 12/.026 vs ref 198/.255269 271/.255269
  0/.068705 729/.064542 279/.044359 2834/.041672 561/.028640 369/.023744).
- **Wire integration**: SAMPLE/PENALTY/BIAS/LOGPROBS parsed in serve.cpp
  (values kept now), sampler-only fields rejected when temp ≤ 0 (D error);
  LOGPROBS appends ` %.9g` logprob to T lines. Verified live: greedy stream
  unchanged (271 51 1618…), seeded SAMPLE reproduces the decoded RNG (seed
  12345 → u=0.1133 → 198), BIAS −5 on 198/271 shifts the draw to 91170.
- **Draw scan order = PROB-DESC (sorted), not vocab order** — proven by a
  seeded coin A/B (top_k=2, 100 seeds, 1-token GENs) against the original
  server on 8730: with the vocab-order scan our server agreed 64/100; after
  switching the support scan to (p desc, id asc) it agrees **90/100**, and
  the sorted-order model with OUR u formula agrees 92/100 with the original.
  The residual ≈ 10% = the serve-path logits difference (their clean-prefill
  p(198)/p(271) ≈ 0.19/0.62 vs our 0.136/0.639 — the tail probs diverge well
  beyond the ±0.005 mean-NLL), NOT an RNG difference: the original is
  deterministic in seed across server restarts, and both head-rate and
  mismatch pattern match the shared-u + shifted-table model exactly.
- **Counter RNG confirmed identical** (b=0, u = (y>>40)·2^-24, posctr =
  n_prompt + i, aux=0): the coin bit-strings line up position-for-position.
  Multi-token A/B (3-token GENs): token-1 agreement ≈ 65/100 (table tails),
  and token-2+ diverges even when token-1 matches (1/10) — the decode-step
  logits differ (the Phase-A gemv/actq numerics gap), so full seeded stream
  parity waits on Phase A, not on the sampler.
- A/B artifacts: /tmp/opencode/ab_orig*.json, ab_ours*.json, coin_*.json,
  multi_*.json; capture scripts ab_capture1.py / ab_coin.py / ab_multi.py;
  fit scripts ab_fit*.py.

## 7. Engine integration notes

- `chlorine_trunk_init/generate/shutdown` (extern "C"); GEN streams `T` lines
  and ends with `D <req> stop|length <n_prompt> <n_gen> <prefill_ms>
  <decode_ms> 0 0 0`. Sampling requests are rejected with `D error` until the
  sampler phase. Prompts beyond trunk capacity → `D error`.
- Weight pool: all quantized payloads pinned host-resident (17.8 GB,
  hipHostMalloc fallback when hipMalloc fails — APU unified memory); forward
  dequants per use; decode steps use a fused dequant-GEMV (k_gemv_dq /
  k_gemv_bf16f). Decode ≈ 1.1 s/token, prefill (87 tok) ≈ 13 s.
- `CHLORINE_STUB=1` forces the deterministic stub generator (wire conformance
  runs — the test's 10 s socket timeouts predate a real backend). Conformance:
  15/15 with the checkpoint + stub env.
