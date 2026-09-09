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

// ================= main =================
int main(int argc, char** argv) {
  const char* mode = argc > 1 ? argv[1] : "tf";
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
