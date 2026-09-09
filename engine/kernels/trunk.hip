// trunk.hip — chlorine-server numeric-core validation harness (hipcc standalone).
// The trunk implementation itself lives in engine/src/generator.hip (shared with
// the server backend); this file only adds the tf/dec validation modes:
//   tf   — teacher-forced replication of the original engine's `--forced` bench
//          (see docs/halogen/TRUNK-NOTES.md for the exact semantics we matched)
//   dec  — prefill + 24 greedy tokens vs tests/equivalence/greedy_text_87.json
// Env knobs: HALO_ACTQ (W4A4 prefill emulation for the forced bench), HALO_DNPREP,
// HALO_BETA32, HALO_BF16LOG, HALO_ROPE, HALO_DEBUG, HALO_STEPS, HALO_TIME,
// HALO_REFDUMP, HALO_TOPK8, HALO_TOPK8_DUMP.
#include "../src/generator.hip"

// ================= sampler self-check =================
// Mirrors the original engine's --sample-check: it runs the FORCED bench
// forward (even-token seq + zero pad = 87 tokens, W4A4 emulation) and samples
// at the LAST row (row 85). Expected table = top-K of the full-vocab softmax,
// renormalized over the support. Reference (original): 198/.255269 271/.255269
// 0/.068705 729/.064542 279/.044359 2834/.041672 561/.028640 369/.023744,
// chi2/df = 0.990.
static const float sc_ref_p[8] = {0.255269f, 0.255269f, 0.068705f, 0.064542f,
                                  0.044359f, 0.041672f, 0.028640f, 0.023744f};
static const int sc_ref_id[8] = {198, 271, 0, 729, 279, 2834, 561, 369};
int samplecheck(int draws, int topk, float temp, float topp, float minp, int row) {
  static float lrow[248320];
  CK(hipMemcpy(lrow, d_logits + (long)row * VOC, (long)VOC * 4, hipMemcpyDeviceToHost));
  // expected table: same filter pipeline as chlorine_sample_host, no RNG
  chlorine_sample_opts o = {};
  o.has_sample = 1; o.temp = temp; o.top_k = topk; o.top_p = topp; o.min_p = minp;
  o.seed = 0;
  static std::vector<int> counts(VOC, 0);
  static std::vector<unsigned char> keep(VOC, 1);
  static std::vector<int> order(VOC);
  static std::vector<float> pr(VOC);
  float mx = lrow[0];
  for (int v = 1; v < VOC; v++) if (lrow[v] > mx) mx = lrow[v];
  float inv_t = 1.0f / temp;
  double tot = 0.0;
  for (int v = 0; v < VOC; v++) { pr[v] = expf((lrow[v] - mx) * inv_t); tot += pr[v]; }
  float ftot = (float)tot;
  for (int v = 0; v < VOC; v++) order[v] = v;
  if (topk > 0 && topk < VOC) {
    std::nth_element(order.data(), order.data() + topk, order.data() + VOC,
                     [&](int a, int b) { return pr[a] != pr[b] ? pr[a] > pr[b] : a < b; });
    std::fill(keep.begin(), keep.end(), 0);
    for (int i = 0; i < topk; i++) keep[order[i]] = 1;
  }
  if (topp > 0.f && topp < 1.f) {
    std::vector<int> nk;
    for (int v = 0; v < VOC; v++) if (keep[v]) nk.push_back(v);
    std::sort(nk.begin(), nk.end(),
              [&](int a, int b) { return pr[a] != pr[b] ? pr[a] > pr[b] : a < b; });
    float cum = 0.f; int cut = (int)nk.size();
    for (int i = 0; i < (int)nk.size(); i++) {
      cum += pr[nk[i]];
      if (cum >= topp * ftot) { cut = i + 1; break; }
    }
    for (int i = cut; i < (int)nk.size(); i++) keep[nk[i]] = 0;
  }
  if (minp > 0.f) {
    float thr = minp / ftot;
    for (int v = 0; v < VOC; v++) if (keep[v] && pr[v] < thr) keep[v] = 0;
  }
  int sup = 0;
  std::vector<std::pair<int, float>> tbl;
  for (int v = 0; v < VOC; v++)
    if (keep[v]) { sup++; tbl.push_back({v, pr[v]}); }
  std::sort(tbl.begin(), tbl.end(),
            [](const std::pair<int, float>& a, const std::pair<int, float>& b) {
              return a.second != b.second ? a.second > b.second : a.first < b.first;
            });
  double kept = 0.0;
  for (auto& e : tbl) kept += e.second;
  printf("sample-check: fixture prompt, K=%d, temp=%.3f, %d draws (row %d)\n", topk, temp, draws, row);
  printf("  filters: top_k=%d top_p=%f min_p=%.4f -> support %d of %d\n",
         topk, topp, minp, sup, topk > 0 && topk < VOC ? topk : VOC);
  // draws
  std::vector<long long> obs(VOC, 0);
  std::vector<int> toks(draws);
  std::vector<float> lps(draws);
  int offsup = 0;
  for (int d = 0; d < draws; d++) {
    int t; float lp;
    chlorine_sample_host(d_logits + (long)row * VOC, counts.data(),
                         (unsigned)(row + d), &o, &t, &lp);
    toks[d] = t; lps[d] = lp;
    obs[t]++;
    if (!keep[t]) offsup++;
  }
  printf("  off-support draws: %d  (MUST be 0 - the sampler may only emit ids the filters kept)\n",
         offsup);
  int shown = getenv("HALO_SC_SHOW") ? atoi(getenv("HALO_SC_SHOW")) : 8;
  for (int r = 0; r < shown && r < (int)tbl.size(); r++) {
    double exp_p = tbl[r].second / kept;
    printf("  %-4d %-10d %.6f %.6f %.1f   ref: %d/%.6f\n", r, tbl[r].first, exp_p,
           (double)obs[tbl[r].first] / draws, exp_p * draws,
           sc_ref_id[r], sc_ref_p[r]);
  }
  double chi2 = 0;
  int cells = 0;
  for (auto& e : tbl) {
    double exp_c = e.second / kept * draws;
    if (exp_c < 5) continue;   // pooled (0 pooled in the reference run)
    double dd = (double)obs[e.first] - exp_c;
    chi2 += dd * dd / exp_c;
    cells++;
  }
  printf("  chi2 = %.2f over df = %d  (%d cell(s) pooled)\n", chi2, cells - 1,
         sup - cells);
  printf("  chi2/df = %.3f  %s\n", chi2 / (cells - 1),
         chi2 / (cells - 1) < 1.6 ? "LOOKS RIGHT" : "OUT OF FAMILY");
  return 0;
}
// ================= main =================
int main(int argc, char** argv) {
  const char* mode = argc > 1 ? argv[1] : "tf";
  if (strcmp(mode, "samplecheck") == 0) {
    if (chlorine_trunk_init(CKPT) != 0) return 1;
    if (!getenv("HALO_ACTQ")) g_actq = 3;      // --sample-check runs the bench path
    if (!getenv("HALO_BETA32")) g_beta32 = 1;
    double t0 = now_s();
    { FILE* pf = fopen("/home/ronin/Projects/chlorine-server/work/opencode/prompt_ids.json", "rb");
      if (!pf) { printf("no prompt file\n"); return 1; }
      static char buf[8192]; size_t n = fread(buf, 1, sizeof(buf) - 1, pf); buf[n] = 0; fclose(pf);
      h_prompt = (int*)malloc(TMAX * 4); h_nprompt = 0;
      for (char* p = buf; *p; p++) {
        if (*p >= '0' && *p <= '9') {
          long v = 0; while (*p >= '0' && *p <= '9') { v = v * 10 + (*p - '0'); p++; }
          h_prompt[h_nprompt++] = (int)v; p--;
        }
      }
    }
    printf("prompt: %d tokens\n", h_nprompt);
    // forced-bench sequence: prompt[0], prompt[2], ... then zero pad = 87
    // tokens (HALO_SC_PLAIN=1: the plain serve prompt + last row instead)
    static int seq[TMAX];
    memset(seq, 0, sizeof(seq));
    int draws = getenv("HALO_DRAWS") ? atoi(getenv("HALO_DRAWS")) : 4000;
    if (getenv("HALO_SC_PLAIN")) {
      memcpy(seq, h_prompt, (size_t)h_nprompt * 4);
      up(d_tok, seq, h_nprompt * 4);
      forward(d_tok, h_nprompt, 0, 1);
      samplecheck(draws, 32, 1.0f, 0.f, 0.f, h_nprompt - 1);
    } else {
      for (int j = 0; 2 * j < h_nprompt; j++) seq[j] = h_prompt[2 * j];
      up(d_tok, seq, 87 * 4);
      forward(d_tok, 87, 0, 1);
      samplecheck(draws, 32, 1.0f, 0.f, 0.f, 85);
    }
    printf("prefill: %.1fs\n", now_s() - t0);
    printf("total elapsed: %.1fs\n", now_s() - t0);
    return 0;
  }
  if (chlorine_trunk_init(CKPT) != 0) return 1;
  if (strcmp(mode, "tf") == 0) {
    // the forced bench runs the engine's W4A4 prefill; emulate unless overridden
    if (!getenv("HALO_ACTQ")) g_actq = 3;
    if (!getenv("HALO_BETA32")) g_beta32 = 1;   // best-known bench emulation config
  }

  double t0 = now_s();
  // ---- prompt ----
  { FILE* pf = fopen("/home/ronin/Projects/chlorine-server/work/opencode/prompt_ids.json", "rb");
    if (!pf) { printf("no prompt file\n"); return 1; }
    static char buf[8192]; size_t n = fread(buf, 1, sizeof(buf) - 1, pf); buf[n] = 0; fclose(pf);
    h_prompt = (int*)malloc(TMAX * 4); h_nprompt = 0;
    for (char* p = buf; *p; p++) {
      if (*p >= '0' && *p <= '9') {
        long v = 0; while (*p >= '0' && *p <= '9') { v = v * 10 + (*p - '0'); p++; }
        h_prompt[h_nprompt++] = (int)v; p--;
      }
    }
    printf("prompt: %d tokens\n", h_nprompt);
  }
  // ---- forced-dump reference (HFD2 dump from the original engine) ----
  // 16-byte header {magic, 0, rows, 1}; rows of 16 B: {u32 pred, u32 tgt, f32 nll, i32 doc}
  int refrows = 0;
  static long refpred[256]; static long reftgt[256]; static float refnll[256];
  { const char* refdump = getenv("HALO_REFDUMP");
    FILE* fd = fopen(refdump ? refdump :
                     "/home/ronin/Projects/chlorine-server/work/opencode/forced-text.bin", "rb");
    if (!fd) { printf("no forced dump\n"); return 1; }
    uint8_t hdr[16]; if (fread(hdr, 1, 16, fd) != 16) { printf("dump hdr\n"); return 1; }
    refrows = hdr[8] | (hdr[9] << 8) | (hdr[10] << 16) | (hdr[11] << 24);
    for (int i = 0; i < refrows; i++) {
      uint8_t r[16]; if (fread(r, 1, 16, fd) != 16) break;
      uint32_t p, t; memcpy(&p, r, 4); memcpy(&t, r + 4, 4);
      float nl2; memcpy(&nl2, r + 8, 4);
      refpred[i] = p; reftgt[i] = t; refnll[i] = nl2;
    }
    fclose(fd);
  }

  // ================= prefill =================
  double tp = now_s();
  int is_tf = (strcmp(mode, "tf") == 0);
  if (is_tf) {
    // The original --forced (FUN_0058c4a0) reads the prompt_ids tensor as an i32
    // vector at 8-BYTE STRIDE (i64 elements): the effective sequence is
    // prompt[0], prompt[2], prompt[4], ... (44 tokens), then zeros past the
    // 348-byte tensor (43 zeros) = 87 tokens. Single window (W=128), teacher
    // forced, ALL 86 next-token rows dumped: row i = position i, tgt = seq[i+1].
    static int seq[TMAX];
    memset(seq, 0, sizeof(seq));
    for (int j = 0; 2 * j < h_nprompt; j++) seq[j] = h_prompt[2 * j];
    up(d_tok, seq, 87 * 4);
    forward(d_tok, 87, 0, 1);
  } else {
    up(d_tok, h_prompt, h_nprompt * 4);
    forward(d_tok, h_nprompt, 0, 1);
  }
  printf("prefill: %.1fs\n", now_s() - tp);

  if (is_tf) {
    static int myrow[256], mytgt[256];
    for (int i = 0; i < refrows; i++) { myrow[i] = i; mytgt[i] = (int)reftgt[i]; }
    up(d_tgt, mytgt, refrows * 4);
    up(d_tok, myrow, refrows * 4);   // reuse d_tok as rowmap
    k_topnll<<<refrows, 256>>>(d_logits, d_tok, d_tgt, d_pred, d_nll, VOC, refrows);
    CK(hipDeviceSynchronize());
    static int mypred[256]; static float mynll[256];
    CK(hipMemcpy(mypred, d_pred, refrows * 4, hipMemcpyDeviceToHost));
    CK(hipMemcpy(mynll, d_nll, refrows * 4, hipMemcpyDeviceToHost));

    int top1 = 0, pmatch = 0, first_pdiv = -1;
    double nllsum = 0, refsum86 = 0;
    unsigned long long hsh = 0x014650fb0739d0383ULL;
    for (int i = 0; i < refrows; i++) {
      refsum86 += refnll[i];
      nllsum += mynll[i];
      if (mypred[i] == mytgt[i]) top1++;
      if (mypred[i] == (int)refpred[i]) pmatch++;
      else if (first_pdiv < 0) first_pdiv = i;
      hsh = (hsh ^ (unsigned)mypred[i]) * 0x100000001b3ULL;
    }
    hsh &= 0xFFFFFFFFFFFFULL;
    printf("== teacher-forced (even-token seq + zero pad, 87 tok, rows @ pos i) ==\n");
    printf("rows=%d predmatch=%d/%d first_pred_div=%d top1=%d/%d (ref 5/86, hash %012llx)\n",
           refrows, pmatch, refrows, first_pdiv, top1, refrows, hsh);
    printf("mean-NLL ours=%.6f ref=%.6f (ground truth 5.918531) delta=%+.6f\n",
           nllsum / refrows, refsum86 / refrows, nllsum / refrows - refsum86 / refrows);
    if (getenv("HALO_TOPK8")) {
      // per-row top-8 ids/logprobs + tail, for comparison vs the HFD3 dump
      static float* rowbuf = (float*)malloc((long)VOC * 4);
      static int eids[256][8]; static float elps[256][8]; static float etail[256];
      { const char* dump = getenv("HALO_TOPK8_DUMP");
        FILE* fd = fopen(dump ? dump : "/tmp/forced-topk8.bin", "rb");
        if (fd) { fseek(fd, 0, SEEK_END); long fsz = ftell(fd); fseek(fd, 0, SEEK_SET);
          std::vector<uint8_t> raw(fsz); if (fread(raw.data(), 1, fsz, fd) == (size_t)fsz) {
            for (int i = 0; i < refrows; i++) {
              const uint8_t* r = raw.data() + 24 + (size_t)i * 84;
              memcpy(eids[i], r + 16, 32); memcpy(elps[i], r + 48, 32); memcpy(&etail[i], r + 80, 4);
            }
          } fclose(fd); }
      }
      for (int i = 0; i < refrows; i++) {
        CK(hipMemcpy(rowbuf, d_logits + (long)myrow[i] * VOC, (long)VOC * 4, hipMemcpyDeviceToHost));
        int ti[8]; float tv[8];
        for (int j = 0; j < 8; j++) { ti[j] = -1; tv[j] = -3.0e38f; }
        double lse = 0; float mx = -3.0e38f;
        for (int v = 0; v < VOC; v++) if (rowbuf[v] > mx) mx = rowbuf[v];
        double sm = 0;
        for (int v = 0; v < VOC; v++) sm += exp((double)rowbuf[v] - mx);
        lse = mx + log(sm);
        for (int v = 0; v < VOC; v++) {
          float x = rowbuf[v];
          if (x > tv[7]) {
            int j = 7; while (j > 0 && x > tv[j - 1]) { tv[j] = tv[j - 1]; ti[j] = ti[j - 1]; j--; }
            tv[j] = x; ti[j] = v;
          }
        }
        double p8 = 0;
        for (int j = 0; j < 8; j++) p8 += exp((double)tv[j] - lse);
        printf("TOPK8 row%2d ours:", i);
        for (int j = 0; j < 8; j++) printf(" %d/%.4f", ti[j], (float)(tv[j] - lse));
        printf(" | tail=%.4f\n", (float)log(1.0 - p8));
        printf("TOPK8 row%2d  ref:", i);
        for (int j = 0; j < 8; j++) printf(" %d/%.4f", eids[i][j], elps[i][j]);
        printf(" | tail=%.4f\n", etail[i]);
      }
    }
    for (int i = 0; i < refrows; i++) {
      if (i < 10 || fabs(mynll[i] - refnll[i]) > 0.05)
        printf("  row%2d pos%3d ours: pred=%6d nll=%8.4f | ref: pred=%6ld tgt=%6ld nll=%8.4f d=%+7.4f\n",
               i, i, mypred[i], mynll[i], refpred[i], reftgt[i], refnll[i], mynll[i] - refnll[i]);
    }
    printf("total elapsed: %.1fs\n", now_s() - t0);
    return 0;
  }

  // ================= greedy decode =================
  static const int target[24] = {271, 51, 1618, 579, 1558, 369, 524, 15756, 264, 13263, 38896, 13,
                                 1049, 369, 279, 15787, 314, 6278, 6165, 7785, 13, 6983, 15019, 3992};
  int gen[24];
  k_argmax1<<<1, 256>>>(d_logits + (long)(h_nprompt - 1) * VOC, d_pred, VOC);
  CK(hipDeviceSynchronize());
  CK(hipMemcpy(&gen[0], d_pred, 4, hipMemcpyDeviceToHost));
  printf("greedy: ");
  printf("%d", gen[0]);
  int matches = 0, gdiv = -1;
  if (gen[0] == target[0]) matches = 1; else gdiv = 0;
  for (int s = 1; s < 24; s++) {
    double td = now_s();
    up(d_tok, &gen[s - 1], 4);
    forward(d_tok, 1, h_nprompt - 1 + s, 1);
    k_argmax1<<<1, 256>>>(d_logits, d_pred, VOC);
    CK(hipDeviceSynchronize());
    CK(hipMemcpy(&gen[s], d_pred, 4, hipMemcpyDeviceToHost));
    if (g_steps) {
      static float lrow[248320];
      CK(hipMemcpy(lrow, d_logits, VOC * 4, hipMemcpyDeviceToHost));
      int t1 = 0, t2 = 0; float v1 = -3e38f, v2 = -3e38f;
      for (int v = 0; v < VOC; v++) {
        if (lrow[v] > v1) { v2 = v1; t2 = t1; v1 = lrow[v]; t1 = v; }
        else if (lrow[v] > v2) { v2 = lrow[v]; t2 = v; }
      }
      printf("  step%d pos%d: top1=%d(%.4f) top2=%d(%.4f) gap=%.4f\n", s, h_nprompt - 1 + s, t1, v1, t2, v2, v1 - v2);
    }
    printf(" %d", gen[s]); fflush(stdout);
    if (getenv("HALO_TIME")) printf("[%.2fs] ", now_s() - td);
    if (gdiv < 0) { if (gen[s] == target[s]) matches++; else gdiv = s; }
  }
  printf("\n== greedy ==\n");
  printf("matches=%d/24 first_div_pos=%d\n", matches, gdiv);
  printf("target: 271 51 1618 579 1558 369 524 15756 264 13263 38896 13 1049 369 279 15787 314 6278 6165 7785 13 6983 15019 3992\n");
  return 0;
}
