# halogen engine — architecture

## Two processes, one container

```
                       container
  ┌────────────────────────────────────────────────────────┐
  │  entrypoint.sh (mode: all|engine|api|bench|sweep)      │
  │                                                        │
  │  ┌──────────────────────────┐   loopback   ┌─────────┐ │
  │  │ /usr/local/bin/halogen   │◄──8730──────►│ python3 │ │────► 0.0.0.0:8731
  │  │ (engine, C++/HIP,        │  token       │ serve_  │ │      (published)
  │  │  gfx1151 kernels)        │  ids only    │ api.py  │ │
  │  └──────────────────────────┘              └─────────┘ │
  │        ▲ mmap: .hgn checkpoint          ▲ /tokenizer   │
  └────────┼────────────────────────────────┼──────────────┘
      -v /models:/models:ro          -v /tokenizer:/tokenizer:ro
```

- The **engine** owns everything tensor-shaped: checkpoint mmap, dequant, the
  trunk (hybrid attention, see below), KV cache, speculation, sampling. It
  speaks a newline-framed ASCII token protocol on 8730 (loopback only in
  `all` mode; **no auth**, kept unpublished by design).
- The **API front-end** (`serve_api.py`) owns everything text-shaped: BPE
  tokenizer, Qwen chat template, OpenAI schemas, SSE framing, tool calls. The
  engine never sees text — only token ids.
- `all` (default) starts the engine, polls the port with bash `/dev/tcp`
  (no fixed sleep — cold 35.9 GB loads fault in slowly), then the API. Either
  process exiting kills the container (a live API in front of a dead engine
  would answer 200 + zero bytes, indistinguishable from a hang).
- `engine` / `api` exist for the two-container compose topology where
  front-end iteration must not cost a model reload.

## Model (from checkpoint names + kernel symbols)

Qwen3.8-27B for AMD Strix Halo (gfx1151), 262144 ctx cap (`kCtxCap`), vocab
248320, hidden 5120 (`0x1400`). Hybrid trunk: DeltaNet (linear attention)
layers plus standard attention layers, an MTP (multi-token prediction) head,
and a DFlash2 n-gram drafter (conv-history based, FP8 codebooks):

- trunk tensors: `embed_tokens.weight`, `layers.*`, `norm.weight`,
  `lm_head.weight`
- MTP head: `mtp.pre_fc_norm_hidden/pre_fc_norm_embedding/fc/fc_out`,
  `mtp.norm.weight`, `mtp.layers.0.*`, `mtp.logits_last`
- DFlash2 drafter: `drafter.fc.weight`, `drafter.chain`,
  `drafter.gate_window`, `drafter.candidate_selector.*`,
  `drafter.alpha_seed`, `drafter.config`, `drafter.config_f32`,
  `draft_vocab_map`, `drafter.hproj`, `successor_codebook`,
  `predecessor_codebook`

## Source units (from `--build-info`)

Host C++: `main.cpp`, `checkpoint.cpp`, `serve.cpp`, `generate.cpp`,
`prefill.cpp`, `spec.cpp`, `drafter.cpp`, `golden.cpp`, `cache_gate.cpp`,
`prompt_cache.cpp`, `matmul.cpp`, `build_check.cpp`
GPU (HIP, gfx1151): `model.hip`, `gemm_i4.hip`, `gemv.hip`, `attn_sd.hip`,
`attn_fa.hip`, `dn_chunk.hip`, `dflash.hip`, `verify.hip`
Build stamp: Sep 5 2026 15:40–41; `kCtxCap=262144`, `kSdLen=683`.

## GPU kernels (59 device symbols in the HIP fatbin)

| Kernel | Purpose (inferred) |
|---|---|
| `k_embed_gather` | embedding lookup |
| `k_dequant<0,1,2>` | dequant paths for the three quant dtypes |
| `k_gemv_ab` | fused GEMV |
| `k_rmsnorm` | RMSNorm |
| `k_actq<0,1>` | activation + quantize (MLP) |
| `k_attn`, `k_attn_fa`, `k_attn_sd_merge` | attention: SDPA path and flash-attention path (env-tunable: `HALOGEN_ATTN_SD`, `HALOGEN_ATTN_FA`, `HALOGEN_FA_OPT`…) |
| `k_dnc_att`, `k_dnc_att2/8/9<0,4,8>`, `k_dnc_ut/ut2/ut3`, `k_dn_step` | DeltaNet chunked attention + state update (chunk size 64, `HALOGEN_DN_CHUNK`) |
| `k_q_scatter` | query/KV scatter into slots |
| `k_conv1d_silu`, `k_conv_hist_update`, `k_conv_hist_commit` | DFlash2 conv history |
| `k_dr_gconv`, `k_dr_qrope`, `k_dr_attn_merge`, `k_dr_topk<16>`, `k_drafter_kv` | drafter forward / candidate top-k |
| `k_accept_prob` | speculative acceptance |
| `k_sample` | sampling (temp/top-k/top-p/min-p/seed) |
| `k_pen_scatter` | frequency/presence penalty |
| `k_add_inplace`, `xor32_kernel` | residual add / counter RNG |

## Serving model

- **Batch-1 serial by default** (one request at a time, one connection held
  behind a lock; second concurrent request → 503 from the front-end).
- **Batched greedy**: `INFO` reports `kv_slots` (env `HALOGEN_KV_SLOTS`) and
  `slot_ctx`; the front-end sizes its semaphore from it. One row per sequence;
  speculation and batched decode cannot share the DeltaNet ring
  (`forward_batch: speculation is live; batched decode and a drafter cannot
  share the DeltaNet ring`).
- **Speculative decoding**: drafter 1 = MTP, drafter 2 = DFlash2 (+MTP verify).
  Byte-identical to serial greedy, verified per release.
- **Prompt cache**: LM-cache of prefill snapshots, auto-sized from
  MemAvailable (`HALOGEN_CACHE_MB` explicit, alignment `HALOGEN_CACHE_ALIGN`),
  reported via `CSTAT`.

## Prompt-cache / KV pool

KV pool printed at startup: `kv pool: %d slot(s) x %d tokens = %.2f GB`.
Cache entries are full-context snapshots (2× a full-context entry is the
floor to enable AUTO cache).
