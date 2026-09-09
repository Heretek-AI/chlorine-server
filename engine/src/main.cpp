// main.cpp — chlorine-server engine entry point.
// Modes this skeleton implements: --checkpoint/--build-info/--serve.
// The generator is a deterministic stub so the wire protocol is testable
// without a GPU; the model backend replaces it in later phases.
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <sys/socket.h>
#include <string>
#include <vector>

#include "hgn.h"
#include "serve.h"

using namespace chlorine;

namespace {

void stub_generate(void* ctx, long id, int max_tokens, const std::vector<int>& eos,
                   const std::vector<int>& prompt, int drafter, bool has_sample,
                   double temp, int top_k, double top_p, double min_p,
                   unsigned long long seed, bool logprobs, int fd) {
  (void)ctx; (void)drafter; (void)has_sample; (void)temp; (void)top_k;
  (void)top_p; (void)min_p; (void)seed;
  // Deterministic stub: token ids derived from the prompt. Never emits an eos
  // unless the prompt itself ends with one (then 0 tokens).
  std::string out;
  char buf[128];
  int n_gen = 0;
  for (int i = 0; i < max_tokens && n_gen < 8; i++) {
    int tok = int((size_t(prompt.empty() ? 0 : prompt.back()) + 7919u * (i + 1)) % 248320u);
    bool stop = false;
    for (int e : eos)
      if (e == tok) { stop = true; break; }
    if (stop) { snprintf(buf, sizeof buf, "D %ld stop %zu %d 0.0 0.0\n", id, prompt.size(), n_gen); out += buf; break; }
    snprintf(buf, sizeof buf, "T %ld %d\n", id, tok);
    out += buf;
    n_gen++;
  }
  if (out.find('D') == std::string::npos) {
    snprintf(buf, sizeof buf, "D %ld length %zu %d 0.1 0.2 0 0 0 0\n", id, prompt.size(), n_gen);
    out += buf;
  }
  size_t off = 0;
  while (off < out.size()) {
    ssize_t n = send(fd, out.data() + off, out.size() - off, MSG_NOSIGNAL);
    if (n <= 0) return;
    off += size_t(n);
  }
}

void build_info(const Checkpoint& ckpt) {
  printf("unit         compiled   kCtxCap   kSdLen\n");
  printf("chlorine     skeleton   262144    683\n");
  printf("checkpoint: %s - %llu tensors, %.1f GB\n", ckpt.model_name().c_str(),
         (unsigned long long)ckpt.n_tensors(), double(ckpt.file_size()) / 1e9);
}

}  // namespace

int main(int argc, char** argv) {
  std::string ckpt_path, bind = "127.0.0.1";
  int port = 8730;
  bool do_serve = false, do_info = false;
  for (int i = 1; i < argc; i++) {
    std::string a = argv[i];
    auto next = [&](const char* what) -> const char* {
      if (i + 1 >= argc) { fprintf(stderr, "%s needs a value\n", what); exit(2); }
      return argv[++i];
    };
    if (a == "--checkpoint") ckpt_path = next("--checkpoint");
    else if (a == "--serve") do_serve = true;
    else if (a == "--port") port = atoi(next("--port"));
    else if (a == "--bind") bind = next("--bind");
    else if (a == "--build-info") do_info = true;
    else { fprintf(stderr, "unknown flag %s\nusage: chlorine --checkpoint FILE.hgn [--build-info] [--serve [--port N] [--bind ADDR]]\n", a.c_str()); return 2; }
  }
  if (!do_serve && !do_info) {
    fprintf(stderr, "usage: chlorine --checkpoint FILE.hgn [--build-info] [--serve [--port N] [--bind ADDR]]\n");
    return 2;
  }
  if (ckpt_path.empty() && do_serve) {
    fprintf(stderr, "--serve needs --checkpoint\n");
    return 2;
  }
  if (ckpt_path.empty()) {
    printf("unit         compiled   kCtxCap   kSdLen\n");
    printf("chlorine     skeleton   262144    683\n");
    if (do_serve) { fprintf(stderr, "--serve needs --checkpoint\n"); return 2; }
    return 0;
  }
  Checkpoint ckpt(ckpt_path);
  if (do_info) build_info(ckpt);
  if (do_serve) {
    Server::Options o;
    o.port = port;
    o.bind = bind;
    o.default_drafter = ckpt.has_dflash2() ? 2 : (ckpt.has_mtp() ? 1 : 0);
    Server svr(ckpt, o, stub_generate, nullptr);
    svr.serve();
  }
  return 0;
}
