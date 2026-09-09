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

1. **i4l weight layout** — per-row sizes resolve as `cols/2 payload + cols/128
   extra` (one scale per 128 columns); scale-block organization unresolved
   (neither per-row tail nor whole-tensor tail as plain e4m3 matches base
   absmax/7). The k_gemm_i4 prologue stages 16 B/thread records with
   `0x90`-byte row steps + group scales — needs the full address-arithmetic
   pass over `work/opencode/gemm0.asm` (~2000 lines) and the LDS/WMMA fragment
   mapping (derivable empirically with a WMMA probe kernel).
2. **Exact accumulation trees** of k_gemv/k_gemm/k_gemm_i4 (per-thread K
   strips, 2-accumulator pattern, xor-butterfly reduce — partially mapped from
   gemv_2_1_1.asm) so bf16 rounding boundaries coincide and the 0.006-margin
   flips resolve. Our fused `k_gemv_dq` mirrors the structure (16-k strips,
   hoisted row scale) but not the exact tree.
3. **k_actq exact scheme** (group size, scale formula, rounding) for the bench
   path.
4. Prompt-cache, spec decoding, KV capacity (trunk context cap is CXT=448
   until the cache grows), per-step perf (q4c GEMV ~1 s/step — vectorize
   dequant, LDS LUT).
5. **Sampler exactness (beyond distribution)**: the k_sample draw-scan
   convention (u×mass target, vocab-order single-thread scan) is our reading
   of the ISA; the top-k selection and the `(int)` conversions around the
   target in `k_sample` (obj1.so @0x19100, two bodies = top-k / no-top-k) are
   not fully decoded. Seeded wire parity vs the original on 8730 is the
   empirical gate.

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
  param1 = weight format {0=q4c-qparam0, 1=q4c-qparam1, 2=fp8r}, PKh arg =
  packed-int4 activations (all GEMMs take quantized activations — there is no
  bf16 GEMM in the engine).
- Serve prefill appears to run the gemv (sub-threshold) route, which is why
  our clean prefill matches the serve stream and not the bench dump.

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
- **Engine tie-break note**: the forced-bench row 85 has 198 and 271 at an
  EXACT bf16 logit tie; the engine's argmax picks 198 (lower id) — k_argmax
  tie order = first max in scan order.

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
