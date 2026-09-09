# halogen wire protocol

Two layers: the engine token protocol (serve.cpp, port 8730) and the OpenAI
API (serve_api.py, port 8731). Sources: engine decompilation of the serve
loop (`FUN_005a9630` GEN handler @0x4a9630; Ghidra base +0x100000) and
`review/halogen-extracted/tools/serve_api.py`.

## Layer 1 — engine protocol (TCP, newline-framed ASCII, NO AUTH)

One connection per client. Requests are one `\n`-terminated line. Replies are
`\n`-terminated lines. The engine only ever sees token ids — the caller does
all tokenization.

### Verbs

| Verb | Grammar | Reply |
|---|---|---|
| `PING` | `PING` | `PONG` |
| `INFO` | `INFO` | one `I` line (see below) |
| `CSTAT` | `CSTAT` | one `C` line — prompt-cache stats |
| `GEN` | see below | stream of `T` lines, then one `D` line |
| `X <req>` | cancel an in-flight request | `D <req> cancel 0 0 0.0 0.0` |

Unknown/malformed lines are dropped (batch-1) — parsing is lenient by design;
the line parser is `istringstream`-based, tokens read with `>>`.

### `INFO` → `I` line

`I %d %d %d %d %d %d %d %lld %d %d %d`

| # | field | meaning (from serve_api.py `_probe`) |
|---|---|---|
| 1 | mtp | MTP head present |
| 2 | draft_head | drafter readout present |
| 3 | ctx | context cap (262144) |
| 4 | spec_rows | verify-batch rows (8) |
| 5 | default | default drafter id |
| 6 | drafter_weights | drafter weights present in checkpoint |
| 7 | dflash2 | drafter 2 actually selectable |
| 8 | cache_mb | prompt-cache cap, MB (0 = off) |
| 9 | cache_align | snapshot alignment |
| 10 | kv_slots | KV slot pool size (batch-1 ⇒ 1) |
| 11 | slot_ctx | per-slot ctx |

An engine that predates a field simply emits a shorter line; the client
treats absence as the conservative default (serial-only, 1 slot).

### `GEN` request

```
GEN <req:i64> <max_tokens:i32> <n_eos:i32> <eos...i32> <n_ids:i32> <ids...i32>
    [<drafter:i32>]                       ; 0 serial / 1 MTP / 2 DFlash2
    [ SAMPLE <temp> <top_k> <top_p> <min_p> <seed> ]   ; %.9g floats, seed u64 REQUIRED for sampling
    [ PENALTY <presence:f32> <frequency:f32> ]
    [ BIAS <n:i32> (<tid:i32> <val:f32>){n} ]          ; n ≤ 20480, tid < 248320
    [ LOGPROBS ]
```

- All three of `PENALTY`/`BIAS`/`LOGPROBS` are **sampler-only**: the engine
  rejects them (error D line) when temperature ≤ 0 — greedy has no sampler to
  apply them to.
- `BIAS` parses as count-then-pairs (engine reads `n`, then `n` id/value
  pairs).
- Drafter the checkpoint can't serve → `D <req> error` (the API surfaces this
  as 400 "engine rejected the request").

### Replies

```
T <req> <token_id>              ; one per emitted token
D <req> <reason> <n_prompt> <n_gen> <prefill_ms> <decode_ms>
D <req> <reason> <n_prompt> <n_gen> <prefill_ms> <decode_ms> <drafter> <rounds> <commit> [n_cached]
D <req> cancel 0 0 0.0 0.0
D <req> error [<n_prompt>] 0 0 0[ 0]
I ...                           ; INFO reply
C %d %lld %lld %lld %ld %ld %ld %ld %ld %.1f %.1f %.1f %.1f %ld
                                ; entries, live bytes, hits, misses, stores,
                                ; evicted, refused, ... , GB figures
```

`reason` ∈ {`done`, `length`, `cancel`, `error`}. With `LOGPROBS`, `T` lines
carry a trailing logprob field (the batched demux tolerates both shapes).
`drafter`/`rounds`/`commit` = speculative stats (serve_api scrapes "rounds"
and "commit" for the bench ledger).

### Batched mode demux

When `kv_slots > 1`, every `T`/`D` line carries `<req>` and a single reader
task fans lines out per request id. A client cancel (`X <req>`) is
fire-and-forget; the eventual `D` line for a gone id is dropped.

### Failure modes (from both sides)

- Engine silent ⇒ front-end budgets: first token 1800 s (prefill, cold 262K
  ≈ 19 min), between tokens 300 s; then disconnect (serial) or fail that
  request (batched).
- Prompt longer than a slot ⇒ `D error`, API → 400.

## Layer 2 — OpenAI-compatible API (8731, FastAPI)

Endpoints: `/v1/models`, `/v1/completions`, `/v1/chat/completions`
(stream=true supported for both), `/health`. Model id reported:
`halogen-qwen3.8-27b`. Errors use OpenAI's `{"error": {message, type, param,
code}}` envelope (not FastAPI's `{"detail"}`).

Notable behaviors in serve_api.py:

- One `Engine` connection, capacity = `kv_slots` from `INFO` (semaphore), all
  writes serialized by a write lock, connects serialized by a separate lock
  (six concurrent reconnects used to leak six sockets).
- Tool calls parsed in `tool_parse.py` (streaming tool-call reassembly,
  message normalization).
- Sampling requires a seed — the engine RNG is counter-based and stateless;
  `%.9g` formatting so `top_p=0.9999999` never rounds to `1` (which would
  disable the nucleus filter engine-side).
- `logprobs`, `logit_bias` (248320 vocab), `presence_penalty`,
  `frequency_penalty`, `temperature`, `top_p`, `seed`, `max_tokens` capped by
  `HALOGEN_MAX_TOKENS_CAP` (65536) and a queue timeout of
  `HALOGEN_QUEUE_TIMEOUT` (7200 s) after which a 503 is returned.
