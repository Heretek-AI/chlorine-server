# halogen 0.1.3 — host logic spec

Reverse-engineered from the decompilation (`review/halogen-extracted/halogen-decomp-all.c`,
683 functions; addresses below are decomp file addresses — vaddr = addr −
0x100000). Line references are `decomp-all.c:N`. Clean-room rebuilds proceed
from THIS SPEC ONLY. `[INFERRED]` marks where the decompiler is ambiguous;
everything else was verified against the decompiled code.

## 1. Entry & dispatch

`main` = `FUN_00559720`. Usage string and mode dispatch at decomp:646/833-930.
Checkpoint loads first (`FUN_005bacf0`), then:

| Mode | Handler | 
|---|---|
| `--verify` | `FUN_0055da50` (gate: verify failure aborts other modes) |
| `--batch-test B gen` | `FUN_005b2540` |
| `--serve port bind` | `FUN_005a3a80` |
| `--cache-gate` | `FUN_005b37a0` |
| `--cache-session` | `FUN_005b5020` |
| `--prefill-bench pp` | `FUN_005a39a0` |
| `--drafter-ingest-bench` | `FUN_005a39d0` |
| `--drafter-round-bench` | `FUN_005a3a00` |
| `--sample-check` | `FUN_0058ee00` |
| `--draft-sample-check` | `FUN_00599580` |
| `--spec-sample-check` | `FUN_0059c460` |
| `--spec-sample-bench` (undocumented) | `FUN_0059dea0` |
| `--forced` | `FUN_0058c4a0` |
| `--mtp-golden` | `FUN_00596730` |
| `--spec` | `FUN_005975a0` |
| `--drafter-taps` | `FUN_0058be60` |
| `--golden` | `FUN_0058fe70` |
| `--build-info` | `FUN_005c0440` (per-TU table via linked list; consistency check `FUN_005c04d0` aborts on mismatched kCtxCap/kSdLen) |

Defaults: port 8730, drafter −1 (unset), draws 20000, sample_k 32,
sample_positions 16, sample_bins 20, pp 512, turns 5, turn_delta 1024,
batch_b 2, temp 1.0.

Original TU map (from `_INIT_*` build stamps): main.cpp, matmul.cpp,
model.hip, attn_fa.hip, attn_sd.hip, dn_chunk.hip, gemm_i4.hip, gemv.hip,
verify.hip, dflash.hip, drafter.cpp, generate.cpp, spec.cpp, prefill.cpp,
prompt_cache.cpp, serve.cpp, checkpoint.cpp, golden.cpp, cache_gate.cpp.
Kernel launches are `__hipPushCallConfiguration` + `hipLaunchKernel` on stubs
registered per TU via `__hipRegisterFunction` (there is no
hipModuleGetFunction).

## 2. Model object (host)

`Model` ctor `FUN_0055e310`. Key fields (offsets on the model object):

| Off | Meaning |
|---|---|
| +0 | n_layers |
| +4 / +5 | has embed_tokens / lm_head (serve refuses without both) |
| +6 / +7 | has_mtp / has_mtp_head buffers |
| +0xc | has_dflash2_weights |
| +0xb8 / +0xc0 | drafter-tap buffer base / row stride |
| +0x110 | DN ring frame (0x9000000 B; ×8 if MTP) |
| +0x118 | conv stash (slots × 0x2d0000) |
| +0x150 | KV pool base (slots × slot_ctx × 0x8000 × 2) |
| +0x156 | pinned current-slot int |
| +0x210 | logits base (row 0x79400 = 248320 × bf16) |
| +0x44c | DN ring cursor |
| +0x460 | conv stash (per-slot) |
| +0x468 | partial-chunk flag (prefill) |
| +0x46c | spec-mode flag (0x101 while MTP drafting) |
| +0x528 / +0x538 | MTP-head KV base / stride |
| +0x540 | KV slot base; +0x548 kv_slots; +0x54c slot_ctx; +0x550 slot stride |
| +0x558 | current slot |
| +0x590 / +0x5b0 | argmax outputs (single / batch) |
| +0x5c0 / +0x5c8 | sampled token / logprob out |
| +0x5d8 / +0x5e8 | residual-sample scratch / atomic counter |
| +0x5f8 | penalty scratch |
| +0x600 / +0x608 / +0x610 | penalty ids / deltas / count (kPenMax 20480) |
| +0x708 / +0x6d8 | MTP logits base / MTP hidden |
| +0x78 / +0x2c | DFlash2 acceptance-seed row / count |

Allocation helper `FUN_00565cd0` accumulates totals. Scratch pool with reuse:
`FUN_00567090` (keyed {tag,size}, cap 0x5dc00001).

KV pool log: `kv pool: %d slot(s) x %d tokens = %.2f GB`. Slots from
`HALOGEN_KV_SLOTS` (1..8), ctx from `HALOGEN_SLOT_CTX` (≤ 0x40000).

## 3. Serve path

Setup `FUN_005a3a80`: socket + SO_REUSEADDR, bind (127.0.0.1 default,
"localhost" accepted; 0.0.0.0 → no-auth warning), listen backlog 0x40,
TCP_NODELAY per fd. Prebuilt INFO line. Mode split at `kv_slots < 2`:

- **Serial loop**: per-connection; line reader `FUN_005a8f30` (memchr '\n',
  select across active + queued fds; queued conn waits, not preempted).
  Pending-fd drain `FUN_005a9310` (MSG_PEEK probe, drops dead sockets).
- **Batched** (`kv_slots ≥ 2`): whole connection handled by `FUN_005a47c0` —
  slot occupancy byte array, GEN line queue, active-generation rows.

Verbs: `PING`→`PONG`; `INFO`→prebuilt I line; `CSTAT`→`FUN_005b0530` + C
line; `GEN <id> ...`→`FUN_005a9630`; `X <id>` (batched)→match against active
rows, send `D <id> cancel 0 0 0.0 0.0`, free row + slot. Unknown → ignored.

INFO line is `snprintf(..., "I %d %d %d %d %d %d %d %lld %d %d %d", has_mtp,
has_mtp_head, 0x40000, 8, default_drafter, has_dflash2_w, has_mtp&has_dflash2,
cache_cap>>20, cache_align, kv_slots, slot_ctx)` — spec_rows is the literal 8.

### GEN handling — `FUN_005a9630`

Parses max_tokens, n_eos + eos ids, n_prompt + prompt ids, then keyed
suffixes (`BIAS` n pairs cap 0x5000; `SAMPLE` temp f32, top_k i32, top_p f32,
min_p f32, seed u64; `PENALTY` 2 f32; `LOGPROBS` flag; any other numeric
token = drafter id).

- max_tokens clamped to `0x3fff7 − n_prompt (+9 if requested 0)` — i.e.
  kCtxCap − 9 headroom (decomp:49751).
- PENALTY/BIAS/LOGPROBS with temp ≤ 0 → warn + `D <id> error 0 0 0 0`.
- Drafter > 2 or unavailable → error D line.
- Then `FUN_005b0920(model, runtime, drafter_bundle, drafter_id, prompt,
  n_prompt, max_tokens, token_sink, cancel_watcher, &stats, sampler_opts)`
  and finally `D %ld %s %d %d %.1f %.1f %d %ld %ld %d` (id, reason, n_prompt,
  max_tokens, prefill_ms, decode_ms, drafter, rounds, commit, cached?).

token_sink (`FUN_005ac6d0`): eos match → reason "stop"; emit `T <id> <tok>`
(+ `" %.6f"` logprob if LOGPROBS); count ≥ max_tokens → "length". In serial
mode a cancel_watcher (`FUN_005acff0`) drains the socket mid-generation and
matches `X <id>` lines → reason "cancel".

### Batched scheduler — `FUN_005a47c0`

- Reads with MSG_DONTWAIT, splits lines, queues GEN lines.
- Scheduling loop: while GEN queue non-empty and a slot byte is 0: claim
  slot, `set_slot` `FUN_00565fe0` (refuses "set_slot: speculation and a
  multi-slot pool both own the DeltaNet ring"), zero slot state `FUN_00565e00`,
  prefill `FUN_0056e890`, first token greedy, emit `T %ld %d`, then per
  scheduler pass advance each active row one token; D line
  `D %ld %s %d %d %.1f 0.0` (step_ms, literal 0.0 decode_ms), free slot.
- Options-carrying GENs (SAMPLE/PENALTY/etc.) run through `FUN_005a9630`
  directly (streaming path) on their own row.
- Teardown `FUN_005ac3e0`: row = {req_id, …, n_gen, start_time}; D line with
  real elapsed; slot byte cleared.

## 4. Generation orchestration — `FUN_005b0920` (generate.cpp)

`caps = has_dflash2*4 + has_mtp*2 + 1`; mode mask per drafter. Sampling active
iff temp>0 and not (top_p==0 && min_p==0 && n_bias==0). BIAS installed via
`FUN_0056f660` (n > 20480 → "over the kPenMax cap").

**Serial (drafter 0)**:
1. Prompt-cache lookup `FUN_005b0100` → matched prefix length (consumed).
2. Prefill remainder `FUN_0056e890` (two-phase when a snapshot boundary
   crosses the prompt: prefill to aligned boundary, store, continue).
3. First token: `FUN_0056ee40` (greedy argmax) or `FUN_0056fa60` (sample).
4. Decode loop: token cb → cancel cb → re-install penalties each round →
   forward 1 token `FUN_0056cb50(model,&tok,0,1,pos)` → sample/greedy.

**Speculative (drafter 1|2)**:
- SpecRunner `FUN_00594180`: `{model, drafter_obj, …}` + 0x2000 token ring;
  MTP sets `model+0x46c = 0x101` so the trunk also produces head taps.
- Prefill + first token `FUN_00594570(out, runner, prompt, n, big, resume,
  snap_at, token_cb, sampler)`: chunked 2048-token prefill; per chunk:
  prefill chunk → final norm `FUN_0056eb80` → **drafter ingest** (vtbl+0x40)
  with 3-token trailing context ring (DFlash2 conv width 3 [INFERRED]); a
  mid-prompt snap_at splits prefill so a snapshot store can happen; first
  token greedy or `FUN_005718b0` (sample_row).
- Round loop: per round:
  - **Adaptive gamma** (gamma<1): u = argmax over 1..max_depth of
    Π accept_rates[2..u+2] / (q_cost + (u−1)·draft_cost + verify_cost).
    Accept rates from the checkpoint's `drafter.alpha_seed` row (each must be
    in (0,1) else fallback defaults 0.809/0.746/0.798/0.772/0.770/0.849).
  - **Draft**: `drafter->vtbl+0x50(last_tok, draft_toks, draft_logprobs,
    gamma, pos, seed)` → n drafts (1..7 enforced).
  - Forward drafts through trunk `FUN_005720c0`; verify probs (sampling:
    `FUN_00570ab0` + k_accept_prob; greedy: host pass).
  - **Host accept loop** (decomp:36542-36640): for i in 0..n−1:
    p = draft_prob[i], q = expf(draft_logprob[i]), threshold min(1, p/q);
    draw u from the **host counter RNG**:
    ```
    c  = ((pos+i) * 0xd1b54a32d192ed03) ^ seed ^ 0x3c6ef372fe94f82a
    c += 0x9e3779b97f4a7c15                      // goldengamma increment
    x  = c; x ^= x >> 30; x *= 0xbf58476d1ce4e5b9
    y  = (x ^ (x >> 27)) * 0x94d049bb133111eb
    u  = (uint)(y >> 36) * 2^-24
    ```
    (splitmix64 finalizer). Accept → emit, continue; reject → break.
  - **Residual correction** on reject at depth d: sample from the normalized
    residual `max(p − q·d, …)` via `FUN_00571280` (k_q_scatter prep +
    k_resid_sample, grid 1×256, args: logits row, pen ids/vals, out tok,
    scratch, atomic, probs, vocab 0x3ca00, temp, top_k, top_p, min_p, seed,
    falsify flag, draft_len 3).
  - All-accepted → continuation token from last verify position.
  - Commit: state advance `FUN_00572290`; drafter update (vtbl+0x60); stats.

DFlash2 vtable `PTR_FUN_005c26a0`: name "dflash2"; max_depth = model+0x2c−1;
state at obj+0x48; ingest requires trunk taps (view+0x10 == model+0xb8)
else exit "dflash2: the trunk view carries no drafter taps"; tap ingest
`FUN_0056db10`; tap capture gated by HALOGEN_ATTN_TAP + HALOGEN_ATTN_TAP_POS.
MTP drafter = stateless singleton `PTR_FUN_005c2718`; entry points
`FUN_00594120` / `FUN_00570000` / `FUN_00570040`; draft math in kernels
(k_drafter_kv shared with DFlash2) [INFERRED].

## 5. Sampler (model.hip host stubs)

1. **Penalties** `FUN_0056f660` install (ids, deltas) → device +0x600/+0x608,
   count +0x610. `FUN_0056f760` = logits_for_row: if penalties, copy row
   (0x79400 B) to +0x5f8 scratch and launch **k_pen_scatter**(scratch, ids,
   deltas, count, 0x3ca00) grid ⌈n/256⌉; else row in place.
2. **Greedy** `FUN_0056ee40` = k_argmax(row, pen_ids, out, vocab, n) grid
   1×256, result from +0x590. Batch variant `FUN_0056f060` (+0x5b0).
3. **Sample** `FUN_0056fa60` (≤128 rows): asserts temp>0 (temperature 0 must
   stay on argmax); k_sample(logits_row, pen_ids, pen_vals, out_tok, out_logp,
   vocab, temp, top_k, top_p, min_p, seed u64, 2×u32) grid (n,1,1)×(256,1,1).
   **top-k, nucleus, min-p, softmax and the draw all happen inside k_sample**;
   penalties beforehand via k_pen_scatter. min_p clamped [0,1] host-side.
4. **Counter RNG**: seed u64 per request; host accept loop derives counters
   `f(pos) ^ seed` (constants above). Kernel-side helper xor32_kernel(state,
   counter, out). `--sample-check` rebuilds the pipeline on host from
   k_topk output to statistically validate k_sample.

`k_accept_prob(target_probs, positions, out, …, floats, ints)` batch-computes
acceptance probabilities. k_resid_sample sig: (logits, probs, out, float*,
vocab, temp, topK, topP, minP, seed, u32, u32, int).

## 6. Prompt cache (prompt_cache.cpp)

Init `FUN_005af230`: HALOGEN_CACHE_MB (0=OFF, −1=auto from MemAvailable −
reserve, reserve HALOGEN_CACHE_RESERVE_MB default 8 GB), HALOGEN_CACHE_ALIGN
(default 0x800=2048; 2048 ⇒ warm decode is bitwise identical to cold, any
other value warns warm output MAY DIFFER). Floor: max(2× full-context entry,
5 GB) else OFF. Entry (64 B): {tokens*, end, ?, mask u32, blob*, aligned_len,
?, stamp(monotonic LRU)}.

- **Lookup** `FUN_005b0100`: only for n_prompt>1 and cap>0. Linear scan,
  longest match wins; match iff entry_len < n_prompt (strict prefix), mask
  superset, and exact token-array prefix compare (no hashes). Hit →
  `FUN_00566930` restore_state: DN ring frame (0x9000000 B) + ring cursor
  reset, conv stash (slot×0x2d0000), 13 KV chunks of len<<12 into the slot's
  KV region (sub-offsets 0x1000…0xf000 × slot_ctx) [INFERRED: per-layer KV
  pages], optional MTP-head KV (mask&2, (len<<12)+0x1000 → +0x700), optional
  DN recurrent state (mask&4: layers×heads×d_state×4 → +0xd0 + 4 B cursor →
  +0x4b0); stats: hits++, matched tokens, ms.
- **Store** `FUN_005b02b0`: skip if an identical full-token entry exists
  (touch only). Make room `FUN_005aff10`: LRU by min stamp; evict while
  live+need > cap or MemAvailable−need < reserve; nothing freeable →
  refused++. Blob via best-fit free-list else hipHostMalloc (pinned host);
  capture `FUN_00566270` = mirror of restore. stores++, bytes accounting.
- Entry blob size: `n*0x10000 + 0x92d0000 (+ (min(n,0x3ffff)<<12)+0x1000 if
  mask&2) (+ dn_state_size+4 if mask&4)` (FUN_00566200).

## 7. Prefill

`FUN_0056e890(model, tokens, count, start)`: validates range within slot
capacity; chunks of 2048; per chunk sets partial-chunk flag `model+0x468`,
then `FUN_0056cb50` forward (tokens or precomputed embeds; position scalar
via k_set_int), then trunk `FUN_0056c1e0`. Fast path when count>0x800 and
HALOGEN_PREFILL_LM and no tap capture: `FUN_0056d190` — chunked
k_embed_gather + batched trunk + final norm + LM head in fewer launches
[INFERRED]. Tap-capture per chunk: `FUN_0056db10` appends trunk rows for the
drafter.

Batched decode entry `FUN_0056ccd0(model, tokens, slots, positions, n)`: n ≤
0x800; refuses when speculation live; rejects duplicate slots ("ONE row per
sequence"); validates slot/capacity bounds.

HALOGEN_DN_CHUNK (default 0x800) sizes the DeltaNet scan chunk and the 4×
0x3000000 scratch workspaces.

## 8. Known-unknowns

- Exact semantics of the 13 KV sub-chunks (per-layer pages) [INFERRED].
- Serial connection-queue capacity and exact multi-step scheduler
  interleaving (loop shape partially obscured; wire formats + slot lifecycle
  exact).
- MTP drafter internal math (kernels only).
- top-p / min-p CLI value sinks (Ghidra lost two stores; sampler opts struct
  confirmed via GEN path).
- Exact per-row counter mix inside k_sample/k_resid_sample (kernel-side).
