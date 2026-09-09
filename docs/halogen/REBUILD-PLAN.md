# chlorine-server — rebuild plan for halogen 0.1.3

Target: a clean-room, AGPL-3.0 reimplementation of the halogen 0.1.3
inference engine, published at `Heretek-AI/chlorine-server`.

Decisions (2026-09-08):

- **Fidelity bar: equivalent replica.** Same wire protocol + OpenAI API, own
  kernels, numerically equivalent — greedy token streams must match the
  original on identical weights; seeded sampling may diverge bitwise unless
  the counter-RNG is mirrored exactly (we will spec it and decide then).
- **Hardware: Strix Halo (gfx1151) box will be available** for validation —
  Framework Desktop / GMKtec EVO-X2 / Beelink GTR9 class (Ryzen AI Max+ 395).
- **License: AGPL-3.0.** Weights never ship with it; the API server is
  network-copyleft under §13 (anyone hosting it must share their changes).

## Legal posture

The Peonist EULA expressly permits reverse engineering ("We do not prohibit
it"; interoperability and security research welcome). §3 prohibits
redistributing a modified image *as halogen* — a from-scratch implementation
is not their Software and sits outside their EULA. They ask that derived
source/kernels not be passed off as one's own work; this plan therefore uses
**clean-room discipline**:

- Reimplementation proceeds from written **behavioral/ISA specs**, never from
  decompiled code pasted in.
- All `review/` material (binary, decompilation, fatbin) stays **out of the
  published repo** — private workspace only.
- Docs may cite observed behavior, formats, and line syntax; they may not
  embed lifted source.

## Current knowledge state (from docs/halogen/, 2026-09-08)

| Asset | Status |
|---|---|
| Engine wire protocol (8730) + OpenAI API (8731) | complete, both sides validated |
| `.hgn` container format | header + 0xa0-stride table + dtype table decoded; qparam byte layouts open |
| Python front-end sources | fully readable (unpublished upstream) |
| CLI + 55 env vars + build/unit map | complete |
| Kernel inventory (59 device symbols) | complete; **kernels are native gfx1151 ISA — no LLVM bitcode in the fatbin** |
| Host internals (scheduler, prefill, spec, sampler, cache) | decompiled, not yet systematically written up |
| Weights (35.9 GB `.hgn`) | not in image; obtainable via HF |
| Golden fixtures | private upstream; we substitute A/B equivalence testing against the original engine |

## Repo layout

```
chlorine-server/
  engine/          C++23 host: checkpoint, serve, generate, prefill, spec,
                   drafter, cache_gate, prompt_cache, matmul, main
  kernels/         HIP gfx1151: model, gemm_i4, gemv, attn_sd, attn_fa,
                   dn_chunk, dflash, verify
  api/             Python OpenAI front-end (FastAPI), clean-room
  converter/       safetensors → .hgn writer + .hgn inspector
  docs/            specs: HOST-LOGIC, KERNELS, CHECKPOINT, PROTOCOL, ENV
  tests/           unit + wire-conformance (CPU-runnable) + A/B equivalence
  deploy/          Dockerfile, compose, entrypoint.sh
  tools/           bench + eval prompts (own implementations)
```

## Phases

### Phase 0 — groundwork (wk 0–1)
Repo scaffold, AGPL-3.0 + notices, gitignore excluding `review/`, Strix Halo
box provisioned with ROCm 7.1x userspace (AMD pip wheels — they ship
clang/comgr, so HIP compilation and gfx1151 ISA disassembly via llvm-mc both
work without a system ROCm install).

### Phase 1 — finish static RE (wk 1–4)
Systematic read of the 683-function decompilation; produce four specs to
`docs/`:

1. **HOST-LOGIC** — scheduler/batching, prefill, KV + DeltaNet ring state,
   speculation orchestration (MTP, DFlash2, adaptive gamma), sampler (incl.
   counter RNG), penalties, prompt-cache eviction.
2. **KERNELS** — all 59 kernels from ISA + vgpr/sgpr/workgroup metadata.
3. **CHECKPOINT** — exact qparam layouts for q4c/fp8r/q8g64/i4l, dims
   semantics, `drafter.config` schema (taps, block_size, sliding_window,
   hidden, q_heads/kv_heads), `draft_vocab_map`, codebooks.
4. **ENV** — the 55 knobs with observed semantics.

Exit criterion: a competent dev can reimplement each kernel from the spec
alone. **No decompiled code enters the repo.**

### Phase 2 — weights pipeline (wk 3–5, overlaps)
Obtain the real `.hgn`; validate the format spec against it with the
inspector; write the safetensors → `.hgn` converter. Weight acquisition is
always the user's step (license separation).

### Phase 3 — host engine rebuild (wk 4–8, overlaps)
C++23, framework-free (matches upstream shape). Wire-protocol conformance
harness runs on CPU with the kernel boundary stubbed. Unit tests for
sampler/scheduler/cache.

### Phase 4 — API front-end rebuild (wk 6–8)
Tokenizer (HF `tokenizers`, Qwen BPE), chat template, SSE, tool-call
assembly; OpenAI-compat suite; response-equality tests against the original
API on the box.

### Phase 5 — GPU kernels (wk 6–14, long pole)
Order: dequant → gemv/gemm_i4 → rmsnorm/act → attn_sd → attn_fa →
dn_chunk (DeltaNet — hardest) → dflash/drafter → verify/accept/sample.
Per kernel: CPU reference → HIP implementation → numeric test vs reference on
the box → integration. Target numeric equivalence, not instruction identity.

### Phase 6 — equivalence + perf validation (wk 14–16)
A/B harness on the box: original vs chlorine-server, identical
weights/prompts/seeds → greedy streams must match byte-for-byte; measure
speculative acceptance deltas; bench vs published figures (57.9 s / 32K
prefill is a stretch goal, not a gate).

### Phase 7 — publish (wk 16)
CI (build + CPU tests in CI, smoke test on self-hosted Strix Halo runner),
docs, Docker packaging, push to `Heretek-AI/chlorine-server`.

## Risk register

| # | Risk | Mitigation |
|---|---|---|
| 1 | Hardware delay | Phases 0–4 are CPU-only and independently useful; kernels start on paper |
| 2 | Quant layout wrong ⇒ divergent output | reverse to the byte in Phase 1; validate against the real `.hgn` in Phase 2 before any kernel is written |
| 3 | DeltaNet chunked update math | most dense kernel family; spec-first, reference-impl-first |
| 4 | Counter-RNG not mirrored ⇒ seeded sampling diverges | spec in Phase 1; if not recoverable, document divergence, keep greedy parity |
| 5 | Perf gap vs hand-tuned W4A4/FA kernels | equivalence gates correctness, not perf; perf is iterative |
| 6 | Upstream changes format in 0.2.x | pin parity target to 0.1.3; version the spec |

## Effort

~4 months part-time, ~2 months full-time with hardware from day one.
Phases 0–4 (~4–6 weeks) yield a protocol/API-compatible engine that can be
merged early and verified incrementally on the box.
