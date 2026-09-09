# halogen engine CLI & environment

From `./halogen --help` and `--build-info` (executed with the image's ROCm
libs; no GPU required for these paths).

## Usage

```
./halogen --checkpoint FILE.hgn [--verify] [--golden FIX.hgn]
          [--mtp-golden MTPFIX.hgn]
          [--forced FIX.hgn [--forced-tokens N] [--forced-dump OUT.bin]]
          [--spec FIX.hgn [--gamma N|0=adaptive] [--gen N] [--depth N] [--drafter 1|2]]
          [--golden FIX.hgn [--drafter-hidden OUT.bin]]
          [--drafter-taps FIX.hgn OUT.bin [--taps-prompt-only]]
          [--prefill-bench FIX.hgn [--pp N]]
          [--drafter-ingest-bench FIX.hgn [--pp N]]
          [--drafter-round-bench FIX.hgn [--pp N]]
          [--sample-check FIX.hgn [--draws N] [--sample-k K] [--temp T]]
          [--draft-sample-check FIX.hgn [--drafter 1|2] [--draws N] [--temp T]
                  [--sample-bins B] [--sample-positions P]]
          [--spec-sample-check FIX.hgn [--drafter 1|2] [--draws N] [--temp T] [--top-p P]]
          [--build-info]
          [--serve [--port N] [--bind ADDR]]
          [--batch-test [--batch-b B] [--gen N]]
          [--cache-gate FIX.hgn [--depth N] [--gen N] [--turn-delta N] [--drafter 0|1|2]]
          [--cache-session FIX.hgn [--depth N] [--turns N] [--turn-delta N] [--gen N]]
```

Modes (exclusive groups): identity/golden fixtures (`--verify`, `--golden`,
`--mtp-golden`, `--forced`), speculation benches (`--spec`, `--drafter-*`),
sampler statistical checks (`--sample-check`, `--draft-sample-check`,
`--spec-sample-check`), serving (`--serve`), batch test, cache tests
(`--cache-gate`, `--cache-session`). The bench/check modes take a "fixture"
checkpoint; `serve` takes the real one.

`--serve` binds loopback by default; binding 0.0.0.0 prints
`serve: WARNING binding 0.0.0.0 - the token protocol has no auth; keep this
port unpublished`.

## Engine environment (55 `HALOGEN_*` vars found in the binary)

### Serving / capacity
`HALOGEN_PORT` `HALOGEN_BIND` `HALOGEN_API_PORT` `HALOGEN_KV_SLOTS`
`HALOGEN_SLOT` `HALOGEN_SLOT_CTX` `HALOGEN_DRAFTER` `HALOGEN_MAX_TOKENS_CAP`
`HALOGEN_QUEUE_TIMEOUT` `HALOGEN_CHECKPOINT` `HALOGEN_TOKENIZER`
`HALOGEN_DOWNLOAD` `HALOGEN_ENGINE`

### Prompt cache
`HALOGEN_CACHE_MB` `HALOGEN_CACHE_RESERVE_MB` `HALOGEN_CACHE_ALIGN`

### Kernel selection / perf switches (documented flavor in FLAGS.md of the
public repo; many are A/B experiment flags)
`HALOGEN_W4A4` `HALOGEN_W4A4_EXCL` `HALOGEN_ATTN_FA` `HALOGEN_ATTN_SD`
`HALOGEN_ATTN_TAP` `HALOGEN_ATTN_TAP_POS` `HALOGEN_FA_OPT` `HALOGEN_FA_PROBE`
`HALOGEN_FA_RS` `HALOGEN_FA_S21` `HALOGEN_FA_TQ` `HALOGEN_FA_VB`
`HALOGEN_DN_ATT_SCALE` `HALOGEN_DN_CHUNK` `HALOGEN_DN_M10G` `HALOGEN_DN_M10P`
`HALOGEN_DN_PROBE` `HALOGEN_DN_S12` `HALOGEN_DN_S14` `HALOGEN_DN_S22`
`HALOGEN_DN_S3A` `HALOGEN_DN_S7` `HALOGEN_DN_SCAN` `HALOGEN_DQ_OVERLAP`
`HALOGEN_PREFILL_DQ` `HALOGEN_PREFILL_LM` `HALOGEN_GEMV_R` `HALOGEN_MATMUL_ALGOS`
`HALOGEN_MATMUL_STATS` `HALOGEN_MATMUL_TUNING_FILE`

### Sampler
`HALOGEN_S2_NUCLEUS` `HALOGEN_S2_QOFF` `HALOGEN_S2_TSKEW` `HALOGEN_S3_FALSIFY`
`HALOGEN_S4_CONV` `HALOGEN_S4_NORM` `HALOGEN_S5_FP32S` `HALOGEN_S11_QKP`
`HALOGEN_S11_R` `HALOGEN_DR_S` `HALOGEN_SD_FIX` `HALOGEN_SD_S` `HALOGEN_SD_V`
`HALOGEN_SERIAL_DUMP` `HALOGEN_SERIAL_REF` `HALOGEN_FORCED_CONT`
`HALOGEN_FORCED_TOPK` `HALOGEN_FORCED_W`

Notes seen in strings: `*_PROBE` modes are "TIMING ONLY, the output of this
run is WRONG by construction"; `HALOGEN_DN_CHUNK` is the DeltaNet chunk size
(64 in the image); `HALOGEN_CACHE_ALIGN` changes prefill chunk boundaries — a
warm answer may differ from a cold one (documented as not covered by identity
tests).

## entrypoint.sh commands (from the image; identical to public repo)

- `all` (default) — engine loopback + API published. One container, one port.
- `engine` / `api` — split topology; engine port must stay unpublished (no
  auth).
- `bench [drafter] [max_tokens] [effort] [reps]` — drives the container's own
  HTTP endpoint (SSE + chat template), ten prompt shapes baked in.
- `sweep …` — llama-bench-shaped pp/tg sweep passed to `tools/halogen-bench.py`.
