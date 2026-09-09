// serve.cpp — engine token protocol. Independent implementation from
// docs/halogen/WIRE-PROTOCOL.md (reversed from the original serve.cpp).
#include "serve.h"

#include <arpa/inet.h>
#include <netinet/in.h>
#include <netinet/tcp.h>
#include <sys/socket.h>
#include <unistd.h>

#include <cerrno>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <iostream>
#include <sstream>
#include <string>
#include <vector>

namespace chlorine {
namespace {

[[noreturn]] void die(const std::string& msg) {
  fprintf(stderr, "%s\n", msg.c_str());
  exit(1);
}

// OpenAI-style error envelope is the API's job; the token protocol just
// reports via the D line.
std::string fmt_d_error(long id) {
  char buf[64];
  snprintf(buf, sizeof buf, "D %ld error 0 0 0 0", id);
  return buf;
}

}  // namespace

Server::Server(const Checkpoint& ckpt, Options opts, GenerateFn gen, void* ctx)
    : ckpt_(ckpt), opts_(opts), gen_(gen), gen_ctx_(ctx) {
  // INFO line — same field order as the original (see WIRE-PROTOCOL.md):
  // I mtp draft_head ctx spec_rows default drafter_weights dflash2
  //   cache_mb cache_align kv_slots slot_ctx
  int mtp = ckpt_.has_mtp() ? 1 : 0;
  int dh = ckpt_.has_mtp() ? 1 : 0;
  int dw = ckpt_.has_dflash2() ? 1 : 0;
  int df2 = (ckpt_.has_mtp() && ckpt_.has_dflash2()) ? 1 : 0;
  int cache_mb = 0, cache_align = 2048;
  char buf[256];
  snprintf(buf, sizeof buf, "I %d %d %d %d %d %d %d %d %d %d %d", mtp, dh,
           opts_.slot_ctx, 8, opts_.default_drafter, dw, df2, cache_mb,
           cache_align, opts_.kv_slots, opts_.slot_ctx);
  info_line_ = buf;
}

void Server::send_all(int fd, const std::string& s) {
  size_t off = 0;
  while (off < s.size()) {
    ssize_t n = ::send(fd, s.data() + off, s.size() - off, MSG_NOSIGNAL);
    if (n <= 0) return;
    off += size_t(n);
  }
}

void Server::handle_gen(int fd, std::istringstream& in) {
  long id = -1;
  if (!(in >> id)) return;
  auto fail = [&](const char* why) {
    fprintf(stderr, "serve: request %ld %s\n", id, why);
    send_all(fd, fmt_d_error(id) + "\n");
  };
  long max_tokens;
  if (!(in >> max_tokens)) return fail("malformed GEN");
  int n_eos;
  if (!(in >> n_eos) || n_eos < 0) return fail("malformed GEN");
  std::vector<int> eos(n_eos);
  for (int i = 0; i < n_eos; i++)
    if (!(in >> eos[i])) return fail("malformed GEN");
  long n_prompt;
  if (!(in >> n_prompt) || n_prompt < 0) return fail("malformed GEN");
  if (n_prompt >= 0x40000) return fail("prompt too long");
  std::vector<int> prompt;
  prompt.resize(size_t(n_prompt));
  for (long i = 0; i < n_prompt; i++)
    if (!(in >> prompt[i])) return fail("malformed GEN");

  int drafter = -1;
  GenOpts o;
  o.has_sample = false;
  std::string tok;
  while (in >> tok) {
    if (tok == "SAMPLE") {
      if (!(in >> o.temp >> o.top_k >> o.top_p >> o.min_p >> o.seed))
        return fail("malformed GEN");
      o.has_sample = true;
    } else if (tok == "PENALTY") {
      if (!(in >> o.presence >> o.frequency)) return fail("malformed GEN");
      if (!o.has_sample) return fail("sent PENALTY/BIAS/LOGPROBS without temperature>0");
    } else if (tok == "BIAS") {
      int n;
      if (!(in >> n)) return fail("malformed GEN");
      if (n < 0 || n > 20480) return fail("malformed GEN");
      o.bias_ids.resize(size_t(n));
      o.bias_vals.resize(size_t(n));
      for (int i = 0; i < n; i++) {
        long long tid;
        double v;
        if (!(in >> tid >> v)) return fail("malformed GEN");
        if (tid < 0 || tid >= 248320) return fail("malformed GEN");
        o.bias_ids[size_t(i)] = int(tid);
        o.bias_vals[size_t(i)] = float(v);
      }
      if (!o.has_sample) return fail("sent PENALTY/BIAS/LOGPROBS without temperature>0");
    } else if (tok == "LOGPROBS") {
      o.logprobs = true;
      if (!o.has_sample) return fail("sent PENALTY/BIAS/LOGPROBS without temperature>0");
    } else {
      drafter = atoi(tok.c_str());
      if (drafter < 0 || drafter > 2) return fail("asked for unknown drafter");
    }
  }
  if (o.has_sample && o.temp <= 0) return fail("SAMPLE requires temperature > 0");
  if (max_tokens < 0) return fail("malformed GEN");
  if (drafter > 2) return fail("asked for unknown drafter");
  long clamp = 0x3fff7 - n_prompt + (max_tokens == 0 ? 9 : 0);
  if (max_tokens > clamp) max_tokens = clamp;
  if (drafter > 0 && drafter == 1 && !ckpt_.has_mtp()) return fail("cannot serve the mtp drafter");
  if (drafter > 0 && drafter == 2 && !ckpt_.has_dflash2()) return fail("cannot serve the dflash2 drafter");
  o.drafter = drafter;
  gen_(gen_ctx_, id, int(max_tokens), eos, prompt, o, fd);
}

void Server::handle_line(int fd, const std::string& line) {
  std::istringstream in(line);
  std::string verb;
  if (!(in >> verb)) return;
  if (verb == "PING") {
    send_all(fd, "PONG\n");
  } else if (verb == "INFO") {
    send_all(fd, info_line_ + "\n");
  } else if (verb == "CSTAT") {
    // cache disabled in the skeleton: entries, bytes, reserved, cap, hits,
    // misses, stores, evicted, tokens_saved, store_ms, restore_ms,
    // last_store_ms, last_restore_ms, refused
    send_all(fd, "C 0 0 0 0 0 0 0 0 0 0.0 0.0 0.0 0.0 0\n");
  } else if (verb == "GEN") {
    handle_gen(fd, in);
  } else if (verb.size() == 1 && verb[0] == 'X') {
    long id;
    in >> id;  // skeleton has no in-flight registry; ignore
  }
  // anything else: ignored (same as the original)
}

void Server::serve() {
  int s = socket(AF_INET, SOCK_STREAM, 0);
  if (s < 0) die("socket");
  int one = 1;
  setsockopt(s, SOL_SOCKET, SO_REUSEADDR, &one, sizeof one);
  sockaddr_in addr {};
  addr.sin_family = AF_INET;
  addr.sin_port = htons(uint16_t(opts_.port));
  if (opts_.bind == "0.0.0.0" || opts_.bind == "::")
    fprintf(stderr,
            "serve: WARNING binding %s - the token protocol has no auth; "
            "keep this port unpublished\n",
            opts_.bind.c_str());
  addr.sin_addr.s_addr = opts_.bind == "localhost" || opts_.bind == "127.0.0.1"
                             ? htonl(INADDR_LOOPBACK)
                             : inet_addr(opts_.bind.c_str());
  if (bind(s, reinterpret_cast<sockaddr*>(&addr), sizeof addr) < 0) die("bind");
  if (listen(s, 64) < 0) die("listen");
  fprintf(stderr, "serve: listening on %s:%d\n", opts_.bind.c_str(), opts_.port);

  std::string buf;
  char rbuf[4096];
  while (true) {
    int fd = accept(s, nullptr, nullptr);
    if (fd < 0) {
      if (errno == EINTR) continue;
      die("accept");
    }
    int nd = 1;
    setsockopt(fd, IPPROTO_TCP, TCP_NODELAY, &nd, sizeof nd);
    buf.clear();
    bool alive = true;
    while (alive) {
      ssize_t n = recv(fd, rbuf, sizeof rbuf, 0);
      if (n <= 0) break;
      buf.append(rbuf, size_t(n));
      size_t pos;
      while ((pos = buf.find('\n')) != std::string::npos) {
        std::string line = buf.substr(0, pos);
        buf.erase(0, pos + 1);
        handle_line(fd, line);
        alive = true;
      }
    }
    close(fd);
  }
}

}  // namespace chlorine
