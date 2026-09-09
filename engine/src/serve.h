// serve.h — the engine token protocol (spec: docs/halogen/WIRE-PROTOCOL.md)
#pragma once
#include <string>
#include <vector>

#include "hgn.h"

namespace chlorine {

// Sampling options carried on the GEN line (wire: SAMPLE/PENALTY/BIAS/LOGPROBS).
struct GenOpts {
  int drafter = 0;
  bool has_sample = false;  // SAMPLE clause present (temp > 0 enforced by parser)
  double temp = 0, top_p = 0, min_p = 0;
  int top_k = 0;
  unsigned long long seed = 0;
  bool logprobs = false;
  double presence = 0, frequency = 0;
  std::vector<int> bias_ids;
  std::vector<float> bias_vals;
};

// One newline-framed ASCII line protocol server. `generate` is the pluggable
// backend; the skeleton ships a deterministic stub so the wire format is
// fully testable without a GPU.
class Server {
 public:
  struct Options {
    int port = 8730;
    std::string bind = "127.0.0.1";
    int default_drafter = -1;
    int kv_slots = 1;
    int slot_ctx = 262144;
    long max_tokens_cap = 65536;
  };

  using GenerateFn = void (*)(void* ctx, long req_id, int max_tokens,
                              const std::vector<int>& eos,
                              const std::vector<int>& prompt, const GenOpts& o,
                              int emit_fd);

  explicit Server(const Checkpoint& ckpt, Options opts, GenerateFn gen, void* ctx);
  void serve();  // blocks forever

 private:
  void handle_line(int fd, const std::string& line);
  void handle_gen(int fd, std::istringstream& in);
  void send_all(int fd, const std::string& s);

  const Checkpoint& ckpt_;
  Options opts_;
  GenerateFn gen_;
  void* gen_ctx_;
  std::string info_line_;
};

}  // namespace chlorine
