# halogen 0.1.3 — reverse-engineering notes

Provenance: `review/halogen.tar`, an OCI image archive of
`ghcr.io/peonist-ai/halogen:0.1.3` (revision `fe8993acaa69`, built 2026-09-05
with buildah 1.43.2). Extracted files live in `review/halogen-extracted/`.

The EULA in the image explicitly permits reverse engineering ("We do not
prohibit it, and we would rather say so"), including reading the gfx1151 device
code with a disassembler.

## What the image actually contains

| Piece | Where | Status vs public repo (`review/halogen-server`) |
|---|---|---|
| Engine binary | `/usr/local/bin/halogen` — 4.99 MB stripped ELF x86-64, AMD clang 23 (ROCm), HIP fatbin for gfx1151 | **not published** — the proprietary core |
| API front-end | `/halogen/tools/serve_api.py` (1701 ln) + `tool_parse.py` (444 ln) | **not published** in the repo |
| Entrypoint | `/usr/local/bin/entrypoint.sh` | identical to `deploy/entrypoint.sh` |
| Bench tools | `halogen-bench.py`, `bench-serving.py`, `eval-prompts.json` | identical to `tools/` |
| Runtime deps | ROCm SDK 7.14 pip wheels + fastapi/uvicorn/transformers stack | N/A |
| Model | `/models/*.hgn` + `/tokenizer` — **not in the image**, volume-mounted | N/A |

Documents in this directory:

- [ARCHITECTURE.md](ARCHITECTURE.md) — process topology, source-unit map, kernel inventory, env knobs
- [WIRE-PROTOCOL.md](WIRE-PROTOCOL.md) — the engine token protocol (port 8730) and the OpenAI API (port 8731)
- [CLI.md](CLI.md) — engine CLI and all 55 `HALOGEN_*` env vars
- [CHECKPOINT-FORMAT.md](CHECKPOINT-FORMAT.md) — the `.hgn` container format
- [HOST-LOGIC.md](HOST-LOGIC.md) — engine host logic spec (dispatch, serve, generation, sampler, prompt cache, prefill)
- [REBUILD-PLAN.md](REBUILD-PLAN.md) — plan to clean-room rebuild this engine (AGPL-3.0) as `Heretek-AI/chlorine-server`

## Method

1. OCI layers extracted from the tar; the four app-bearing layers are the last
   small ones (`da2f12c1` binary, `90ee0325` tools, `8319ab7b` entrypoint,
   `fa27f532` licenses).
2. Python front-end read directly (`review/halogen-extracted/tools/`).
3. `--build-info` and `--help` executed under the image's own ROCm libs
   (LD_LIBRARY_PATH into the extracted site-packages) — works with no GPU.
4. Ghidra headless decompilation of all 683 functions (PIE base 0x100000;
   Ghidra addresses = vaddr + 0x100000), cross-checked against objdump.
5. Wire protocol cross-validated between the binary and `serve_api.py`, which
   documents both sides of every line format.
