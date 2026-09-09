# chlorine-server

An independent, AGPL-3.0 reimplementation of the halogen 0.1.3 inference
engine for **Qwen3.8-27B on AMD Strix Halo (gfx1151)** — clean-room rebuilt
from reverse-engineering notes in `docs/halogen/`.

- Same wire protocol as halogen's engine (`PING/INFO/CSTAT/GEN/X` on the
  token port) and the same OpenAI-compatible serving API on `:8731`.
- Loads `.hgn` checkpoints (see `converter/` and `docs/halogen/CHECKPOINT-FORMAT.md`).
- Greedy token streams match the original engine on identical weights.

**Not** affiliated with Peonist or halogen. The original engine is closed
source and lives at [peonist-ai/halogen-server](https://github.com/peonist-ai/halogen-server);
its EULA permits reverse engineering and interoperability. This codebase is
written from behavioral specifications only.

## Status

Work in progress — see [docs/halogen/REBUILD-PLAN.md](docs/halogen/REBUILD-PLAN.md).

| Component | State |
|---|---|
| `docs/halogen/` RE notes + specs | in progress |
| `engine/` C++23 host | scaffolded |
| `engine/kernels/` HIP gfx1151 | scaffolded |
| `api/` OpenAI front-end | scaffolded |
| `converter/` safetensors → `.hgn` | scaffolded |

## Weights

Never in this repo. Bring your own: the halogen `.hgn` checkpoints are
published separately (`peonist-ai/halogen-qwen3.8-27b` on Hugging Face), or
convert your own Qwen3.8-27B safetensors with `converter/`.

## License

AGPL-3.0 — see [LICENSE](LICENSE). Model weights are governed by their own
licenses and are explicitly excluded from this repository's terms.
