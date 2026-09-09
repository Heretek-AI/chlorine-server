// trunk.hip — chlorine-server numeric core validation (spec: docs/halogen/*.md)
// Qwen3.8-27B hybrid trunk (48 GDN + 16 FA layers) on gfx1151.
// Math source: HF transformers Qwen3.5 modeling (Apache-2.0) — GatedDeltaNet,
// Qwen3_5Attention, Qwen3_5RMSNorm(ZeroCentered), Qwen3_5RMSNormGated.
// Weight formats: docs/halogen/KERNELS-DEQUANT.md (validated formulas reused).
//
// Modes:  tf   — teacher-forced forward over /tmp/prompt_ids.json (87 tokens),
//                compare vs /tmp/forced-text.bin (HFD2 dump from the original engine)
//         dec  — prefill + 24 greedy tokens, compare vs the original's stream
#include <hip/hip_runtime.h>
#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <vector>
#include <map>
#include <algorithm>
#include <sys/time.h>

#define CK(x) do { hipError_t e = (x); if (e) { printf("HIP err %s @%d\n", hipGetErrorString(e), __LINE__); exit(1);} } while (0)
#define GRID1(n) (unsigned)((((long)(n)) + 255) / 256)

// ---------------- model constants ----------------
static const int   HID   = 5120;      // hidden
static const int   VOC   = 248320;    // vocab
static const int   NL    = 64;        // layers; layer%4==3 -> full_attention
static const int   INTM  = 17408;     // mlp intermediate
// GDN (linear_attn)
static const int   HK    = 16;        // k heads
static const int   HV    = 48;        // v heads
static const int   DK    = 128;       // k head dim
static const int   DV    = 128;       // v head dim
static const int   QKVW  = 10240;     // qkv width (q 2048 | k 2048 | v 6144)
// FA (self_attn)
static const int   QH    = 24;        // q heads
static const int   KVH   = 4;         // kv heads
static const int   HD    = 256;       // head dim
static const int   NPAIR = 32;        // rotary pairs (partial 0.25 -> 64 dims)
static const float SCALE = 0.0625f;   // 1/sqrt(256)
static const float THETA = 1e7f;
static const float EPS   = 1e-6f;
static const int   CXT   = 448;      // kv cache capacity (>= 348 u16-seq prefill + 24 decode)
static const int   TMAX  = 352;      // activation rows (>= 348 forced-seq)
static const char* CKPT  = "/home/ronin/Projects/chlorine-server/models/qwen3.8-27b-p1w4d-d2.hgn";

static int ROPE_MODE = 0;   // 0: HF half-split pairs (p, p+32); 1: interleaved (2p, 2p+1)

static double now_s() { struct timeval tv; gettimeofday(&tv, nullptr); return tv.tv_sec + tv.tv_usec * 1e-6; }

// ---------------- bf16 helpers ----------------
__device__ inline float bf16f(uint16_t h) { uint32_t u = (uint32_t)h << 16; return __uint_as_float(u); }
__device__ inline uint16_t f32bf(float f) {
  uint32_t u = __float_as_uint(f);
  uint32_t lo = u & 0xFFFFu;
  u = (u >> 16) + ((lo > 0x7FFFu) | ((lo == 0x7FFFu) & ((u >> 16) & 1u)));
  return (uint16_t)(u & 0xFFFFu);
}
static inline float h_bf16f(uint16_t h) { uint32_t u = (uint32_t)h << 16; float f; memcpy(&f, &u, 4); return f; }
static inline uint16_t h_f32bf(float f) {
  uint32_t u; memcpy(&u, &f, 4);
  uint32_t lo = u & 0xFFFFu;
  u = (u >> 16) + ((lo > 0x7FFFu) | ((lo == 0x7FFFu) & ((u >> 16) & 1u)));
  return (uint16_t)(u & 0xFFFFu);
}

// ---------------- dequant (from dequant_test.hip, validated) ----------------
__device__ inline float dequant_fp8_e4m3(uint32_t b) {
  float s = (b & 0x80u) ? -1.0f : 1.0f;
  uint32_t e = (b >> 3) & 0xFu, m = b & 0x7u;
  if (e == 0) return s * (float)m * 0.125f * ldexpf(1.0f, -6);
  return s * (1.0f + (float)m * 0.125f) * ldexpf(1.0f, (int)e - 7);
}
__global__ void k_dequant_fp8r(const uint8_t* __restrict__ payload, const uint16_t* __restrict__ scales,
                               uint16_t* __restrict__ out, long cols) {
  long i = blockIdx.x * (long)blockDim.x + threadIdx.x;
  long r = i / cols;
  float w = dequant_fp8_e4m3(payload[i]) * bf16f(scales[r]);
  out[i] = f32bf(w);
}
__global__ void k_dequant_q4c(const float* __restrict__ cb, const uint8_t* __restrict__ rowbase,
                              uint16_t* __restrict__ out, long cols) {
  long i = blockIdx.x * (long)blockDim.x + threadIdx.x;
  long r = i / cols, c = i % cols;
  const uint8_t* pay = rowbase + (long)r * (cols / 2 + cols / 16);
  uint8_t byte = pay[c / 2];
  uint32_t nib = (c & 1) ? ((byte >> 4) & 0xFu) : (byte & 0xFu);
  float w = cb[nib] * dequant_fp8_e4m3(pay[cols / 2 + c / 16]);
  out[i] = f32bf(w);
}

// ---------------- GEMM: C[M,N] = A[M,K] x W[N,K]^T, bf16 in, f32 accum ----------------
// block 16x16; grid.x = ceil(N/64); block loops all 16-row m-tiles (M <= 96).
template<int F32OUT>
__global__ void k_gemm(const uint16_t* __restrict__ A, const uint16_t* __restrict__ W,
                       uint16_t* __restrict__ C, float* __restrict__ Cf,
                       int M, int N, int K) {
  int n0 = blockIdx.x * 64;
  int tx = threadIdx.x, ty = threadIdx.y;
  __shared__ uint16_t sw[64][64];
  __shared__ uint16_t sa[16][64];
  float acc[6][4];
  for (int i = 0; i < 6; i++) for (int j = 0; j < 4; j++) acc[i][j] = 0.f;
  for (int k0 = 0; k0 < K; k0 += 64) {
    for (int s = 0; s < 16; s++) {
      int l = (ty * 16 + tx) + s * 256;
      int r = l >> 6, c = l & 63;
      long n = (long)n0 + r; long k = (long)k0 + c;
      sw[r][c] = (n < N && k < K) ? W[n * K + k] : (uint16_t)0;
    }
    __syncthreads();
    for (int mt = 0; mt < 6; mt++) {
      int m0 = mt * 16;
      for (int s = 0; s < 4; s++) {
        int c = tx + s * 16;
        long k = (long)k0 + c; int m = m0 + ty;
        sa[ty][c] = (m < M && k < K) ? A[(long)m * K + k] : (uint16_t)0;
      }
      __syncthreads();
      for (int kk = 0; kk < 64; kk++) {
        float av = bf16f(sa[ty][kk]);
        acc[mt][0] += av * bf16f(sw[tx][kk]);
        acc[mt][1] += av * bf16f(sw[tx + 16][kk]);
        acc[mt][2] += av * bf16f(sw[tx + 32][kk]);
        acc[mt][3] += av * bf16f(sw[tx + 48][kk]);
      }
      __syncthreads();
    }
  }
  for (int mt = 0; mt < 6; mt++) {
    int m = mt * 16 + ty;
    if (m >= M) continue;
    for (int e = 0; e < 4; e++) {
      long n = (long)n0 + tx + e * 16;
      if (n >= N) continue;
      if (F32OUT) Cf[(long)m * N + n] = acc[mt][e];
      else C[(long)m * N + n] = f32bf(acc[mt][e]);
    }
  }
}
static void dumpx(const uint16_t* d_x) {
  static int xi = 0; static FILE* xf = 0;
  const char* dxp = getenv("HALO_DUMP_X");
  if (!dxp) return;
  if (!xf) xf = fopen(dxp, "wb");
  void* hb = malloc(512);
  CK(hipMemcpy(hb, d_x, 512, hipMemcpyDeviceToHost));
  int idx = xi++;
  fwrite(&idx, 4, 1, xf);
  fwrite(hb, 1, 512, xf);
  free(hb);
}
static void gemm(const uint16_t* A, const uint16_t* W, uint16_t* C, int M, int N, int K) {
  dim3 b(16, 16);
  for (int m0 = 0; m0 < M; m0 += 96) {          // kernel supports M <= 96; chunk by m
    int mm = M - m0 < 96 ? M - m0 : 96;
    dim3 g((N + 63) / 64);
    hipLaunchKernelGGL((k_gemm<0>), g, b, 0, 0, A + (long)m0 * K, W, C + (long)m0 * N, (float*)nullptr, mm, N, K);
  }
}
static void gemm_f32(const uint16_t* A, const uint16_t* W, float* Cf, int M, int N, int K) {
  dim3 b(16, 16);
  for (int m0 = 0; m0 < M; m0 += 96) {
    int mm = M - m0 < 96 ? M - m0 : 96;
    dim3 g((N + 63) / 64);
    hipLaunchKernelGGL((k_gemm<1>), g, b, 0, 0, A + (long)m0 * K, W, (uint16_t*)nullptr, Cf + (long)m0 * N, mm, N, K);
  }
}

// ---------------- norms ----------------
__global__ void k_rmsnorm(const uint16_t* __restrict__ X, const uint16_t* __restrict__ w,
                          uint16_t* __restrict__ O, int K, int T, int zero_center) {
  int row = blockIdx.x; if (row >= T) return;
  __shared__ float sh[256];
  int tid = threadIdx.x;
  float s = 0.f;
  for (int c = tid; c < K; c += 256) { float v = bf16f(X[(long)row * K + c]); s += v * v; }
  sh[tid] = s; __syncthreads();
  for (int off = 128; off > 0; off >>= 1) { if (tid < off) sh[tid] += sh[tid + off]; __syncthreads(); }
  float inv = rsqrtf(sh[0] / (float)K + EPS);
  for (int c = tid; c < K; c += 256) {
    float v = bf16f(X[(long)row * K + c]);
    float wv = zero_center ? (1.0f + bf16f(w[c])) : bf16f(w[c]);
    O[(long)row * K + c] = f32bf(v * inv * wv);
  }
}
// per-head RMSNorm (zero-centered (1+w)): src row stride sstride, head h at offset
// h*hstride (FA q_proj rows interleave [q_h | gate_h], so q needs hstride=2*D); dst [T,heads,D]
__global__ void k_headnorm(const uint16_t* __restrict__ X, const uint16_t* __restrict__ w,
                           uint16_t* __restrict__ O, int T, int heads, int sstride, int hstride, int D) {
  long i = blockIdx.x * (long)blockDim.x + threadIdx.x;
  if (i >= (long)T * heads) return;
  int t = (int)(i / heads), h = (int)(i % heads);
  const uint16_t* xp = X + (long)t * sstride + (long)h * hstride;
  float ss = 0.f;
  for (int d = 0; d < D; d++) { float v = bf16f(xp[d]); ss += v * v; }
  float inv = rsqrtf(ss / (float)D + EPS);
  uint16_t* op = O + ((long)t * heads + h) * D;
  for (int d = 0; d < D; d++) {
    float nrm = bf16f(xp[d]) * inv;
    op[d] = f32bf(nrm * (1.0f + bf16f(w[d])));
  }
}

// ---------------- embed gather / residual ----------------
__global__ void k_embed(const int* __restrict__ ids, const uint16_t* __restrict__ E,
                        uint16_t* __restrict__ X, int H) {
  long i = blockIdx.x * (long)blockDim.x + threadIdx.x;
  long t = i / H;
  X[i] = E[(long)ids[t] * H + (i % H)];
}
__global__ void k_add(const uint16_t* __restrict__ A, const uint16_t* __restrict__ B,
                      uint16_t* __restrict__ O, long n) {
  long i = blockIdx.x * (long)blockDim.x + threadIdx.x;
  if (i >= n) return;
  O[i] = f32bf(bf16f(A[i]) + bf16f(B[i]));
}

// ---------------- conv1d (causal, kernel 4, groups, silu) ----------------
__global__ void k_conv1d(const uint16_t* __restrict__ X, const uint16_t* __restrict__ Wc,
                         uint16_t* __restrict__ O, long T) {
  long i = blockIdx.x * (long)blockDim.x + threadIdx.x;
  if (i >= T * QKVW) return;
  int t = (int)(i / QKVW), c = (int)(i % QKVW);
  float acc = 0.f;
  for (int j = 0; j < 4; j++) {
    int tt = t + j - 3;
    if (tt >= 0) acc += bf16f(Wc[c * 4 + j]) * bf16f(X[(long)tt * QKVW + c]);
  }
  uint16_t yb = f32bf(acc);
  float yf = bf16f(yb);
  O[i] = f32bf(yf / (1.0f + expf(-yf)));
}
__global__ void k_conv1d_step(const uint16_t* __restrict__ X, const uint16_t* __restrict__ Wc,
                              uint16_t* __restrict__ O, uint16_t* __restrict__ st) {
  int c = blockIdx.x * blockDim.x + threadIdx.x;
  if (c >= QKVW) return;
  float acc = bf16f(st[c * 3 + 0]) * bf16f(Wc[c * 4 + 0])
            + bf16f(st[c * 3 + 1]) * bf16f(Wc[c * 4 + 1])
            + bf16f(st[c * 3 + 2]) * bf16f(Wc[c * 4 + 2])
            + bf16f(X[c])          * bf16f(Wc[c * 4 + 3]);
  uint16_t yb = f32bf(acc);
  float yf = bf16f(yb);
  O[c] = f32bf(yf / (1.0f + expf(-yf)));
  st[c * 3 + 0] = st[c * 3 + 1];
  st[c * 3 + 1] = st[c * 3 + 2];
  st[c * 3 + 2] = X[c];
}
__global__ void k_conv_setstate(const uint16_t* __restrict__ X, uint16_t* __restrict__ st, long T) {
  int c = blockIdx.x * blockDim.x + threadIdx.x;
  if (c >= QKVW) return;
  for (int j = 0; j < 3; j++) {
    long tt = T - 3 + j;
    st[c * 3 + j] = (tt >= 0) ? X[tt * QKVW + c] : (uint16_t)0;   // zero pad (prefill T < 3)
  }
}

// ---------------- RoPE (partial: 64 dims of 256, theta 1e7) ----------------
__global__ void k_rope(uint16_t* __restrict__ X, long T, int heads, int pos_base, int mode) {
  long i = blockIdx.x * (long)blockDim.x + threadIdx.x;
  long total = (long)T * heads * NPAIR;
  if (i >= total) return;
  int p = (int)(i % NPAIR);
  int h = (int)((i / NPAIR) % heads);
  int t = (int)(i / ((long)NPAIR * heads));
  uint16_t* xp = X + ((long)t * heads + h) * HD;
  int pos = pos_base + t;
  float inv = powf(THETA, -(float)(2 * p) / 64.0f);
  float ang = (float)pos * inv;
  float c = cosf(ang), s = sinf(ang);
  uint16_t cb = f32bf(c), sb = f32bf(s);   // HF casts cos/sin to bf16 before applying
  float cf = bf16f(cb), sf = bf16f(sb);
  int d1, d2;
  if (mode) { d1 = 2 * p; d2 = 2 * p + 1; } else { d1 = p; d2 = p + NPAIR; }
  float a1 = bf16f(xp[d1]), a2 = bf16f(xp[d2]);
  // HF: (q_rot*cos) + (rotate_half(q_rot)*sin), each product and the sum rounded to bf16
  float t1 = bf16f(f32bf(a1 * cf)), t2 = bf16f(f32bf(a2 * sf));
  float t3 = bf16f(f32bf(a2 * cf)), t4 = bf16f(f32bf(a1 * sf));
  xp[d1] = f32bf(t1 - t2);
  xp[d2] = f32bf(t3 + t4);
}

// ---------------- KV cache write ----------------
__global__ void k_kvwrite(const uint16_t* __restrict__ S, uint16_t* __restrict__ KC,
                          long T, int pos_base) {
  long i = blockIdx.x * (long)blockDim.x + threadIdx.x;
  if (i >= (long)T * KVH * HD) return;
  int d = (int)(i % HD), h = (int)((i / HD) % KVH), t = (int)(i / ((long)KVH * HD));
  KC[((long)(pos_base + t) * KVH + h) * HD + d] = S[i];
}

// ---------------- FA attention ----------------
__global__ void k_attn_scores(const uint16_t* __restrict__ Q, const uint16_t* __restrict__ KC,
                              float* __restrict__ S, int T, int Tmax) {
  long i = blockIdx.x * (long)blockDim.x + threadIdx.x;
  if (i >= (long)QH * T * T) return;
  int j = (int)(i % T), t = (int)((i / T) % T), h = (int)(i / ((long)T * T));
  if (j > t) return;
  int kvh = h / (QH / KVH);
  const uint16_t* qp = Q + ((long)t * QH + h) * HD;
  const uint16_t* kp = KC + ((long)j * KVH + kvh) * HD;
  float s = 0.f;
  for (int d = 0; d < HD; d++) s += bf16f(qp[d]) * bf16f(kp[d]);
  S[(long)h * Tmax * Tmax + (long)t * Tmax + j] = s * SCALE;
}
__global__ void k_attn_softmax(const float* __restrict__ S, uint16_t* __restrict__ Pbf, int T, int Tmax) {
  long i = blockIdx.x * (long)blockDim.x + threadIdx.x;
  if (i >= (long)QH * T) return;
  int t = (int)(i % T), h = (int)(i / T);
  const float* row = S + (long)h * Tmax * Tmax + (long)t * Tmax;
  uint16_t* pb = Pbf + (long)h * Tmax * Tmax + (long)t * Tmax;
  float mx = -3.0e38f;
  for (int j = 0; j <= t; j++) mx = fmaxf(mx, row[j]);
  float sum = 0.f;
  for (int j = 0; j <= t; j++) sum += expf(row[j] - mx);
  float inv = 1.0f / sum;
  for (int j = 0; j <= t; j++) pb[j] = f32bf(expf(row[j] - mx) * inv);
}
__global__ void k_attn_pv(const uint16_t* __restrict__ Pbf, const uint16_t* __restrict__ VC,
                          uint16_t* __restrict__ O, int T, int Tmax) {
  long i = blockIdx.x * (long)blockDim.x + threadIdx.x;
  if (i >= (long)T * QH * HD) return;
  int d = (int)(i % HD), h = (int)((i / HD) % QH), t = (int)(i / ((long)QH * HD));
  int kvh = h / (QH / KVH);
  const uint16_t* pb = Pbf + (long)h * Tmax * Tmax + (long)t * Tmax;
  float acc = 0.f;
  for (int j = 0; j <= t; j++) acc += bf16f(pb[j]) * bf16f(VC[((long)j * KVH + kvh) * HD + d]);
  O[((long)t * QH + h) * HD + d] = f32bf(acc);
}
__global__ void k_attn_step(const uint16_t* __restrict__ Qrow, const uint16_t* __restrict__ KC,
                            const uint16_t* __restrict__ VC, uint16_t* __restrict__ Orow, int pos) {
  int h = blockIdx.x; if (h >= QH) return;
  __shared__ float sp[CXT + 1];
  __shared__ uint16_t sb[CXT + 1];
  int tid = threadIdx.x;
  int kvh = h / (QH / KVH);
  if (tid <= pos) {
    const uint16_t* qp = Qrow + h * HD;
    const uint16_t* kp = KC + ((long)tid * KVH + kvh) * HD;
    float s = 0.f;
    for (int d = 0; d < HD; d++) s += bf16f(qp[d]) * bf16f(kp[d]);
    sp[tid] = s * SCALE;
  }
  __syncthreads();
  if (tid == 0) {
    float mx = -3.0e38f;
    for (int j = 0; j <= pos; j++) mx = fmaxf(mx, sp[j]);
    float sum = 0.f;
    for (int j = 0; j <= pos; j++) { float e = expf(sp[j] - mx); sp[j] = e; sum += e; }
    float inv = 1.0f / sum;
    for (int j = 0; j <= pos; j++) sb[j] = f32bf(sp[j] * inv);
  }
  __syncthreads();
  if (tid < HD) {
    float acc = 0.f;
    for (int j = 0; j <= pos; j++) acc += bf16f(sb[j]) * bf16f(VC[((long)j * KVH + kvh) * HD + tid]);
    Orow[h * HD + tid] = f32bf(acc);
  }
}

// ---------------- FA output gate ----------------
__global__ void k_fa_gate(uint16_t* __restrict__ A, const uint16_t* __restrict__ QG, long T) {
  long i = blockIdx.x * (long)blockDim.x + threadIdx.x;
  if (i >= (long)T * QH * HD) return;
  int d = (int)(i % HD), h = (int)((i / HD) % QH), t = (int)(i / ((long)QH * HD));
  float g = bf16f(QG[(long)t * (QH * HD * 2) + (long)h * HD * 2 + HD + d]);
  float sg = 1.0f / (1.0f + expf(-g));
  A[i] = f32bf(bf16f(A[i]) * bf16f(f32bf(sg)));
}

// ---------------- GDN output gated RMSNorm ----------------
__global__ void k_gdn_gatednorm(const uint16_t* __restrict__ CO, const uint16_t* __restrict__ Z,
                                const uint16_t* __restrict__ w, uint16_t* __restrict__ O, long T) {
  long i = blockIdx.x * (long)blockDim.x + threadIdx.x;
  if (i >= (long)T * HV) return;
  int t = (int)(i / HV), h = (int)(i % HV);
  const uint16_t* hp = CO + (long)t * (HV * DV) + h * DV;
  const uint16_t* zp = Z + (long)t * (HV * DV) + h * DV;
  float ss = 0.f;
  for (int d = 0; d < DV; d++) { float v = bf16f(hp[d]); ss += v * v; }
  float inv = rsqrtf(ss / (float)DV + EPS);
  for (int d = 0; d < DV; d++) {
    float hn = bf16f(hp[d]) * inv;
    uint16_t hb = f32bf(hn);                 // f32 norm -> bf16 RNE, then weight (HF order)
    float m = bf16f(hb) * bf16f(w[d]);       // RMSNormGated: direct w (not zero-centered)
    uint16_t mb = f32bf(m);
    float z = bf16f(zp[d]);
    float sg = z / (1.0f + expf(-z));        // silu(z) in f32
    O[(long)t * (HV * DV) + h * DV + d] = f32bf(bf16f(mb) * sg);
  }
}

// ---------------- MLP act ----------------
// engine-default W4A4: activations quantized before the q4c GEMMs (k_actq).
// Emulate: symmetric int4 grid, per-row bf16 scale, RNE round, then dequant to bf16.
__global__ void k_actq(uint16_t* __restrict__ X, int K, int mode) {
  // one block per row
  __shared__ float sh[256];
  int tid = threadIdx.x;
  const uint16_t* row = X + (long)blockIdx.x * K;
  float am = 0.f;
  for (int c = tid; c < K; c += 256) { float v = bf16f(row[c]); am = fmaxf(am, fabsf(v)); }
  sh[tid] = am; __syncthreads();
  for (int off = 128; off > 0; off >>= 1) { if (tid < off) sh[tid] = fmaxf(sh[tid], sh[tid + off]); __syncthreads(); }
  float scale = bf16f(f32bf(sh[0] / 7.0f));
  if (scale <= 0.f) scale = 1.0f;
  uint16_t* orow = X + (long)blockIdx.x * K;
  for (int c = tid; c < K; c += 256) {
    float v = bf16f(orow[c]);
    float q = rintf(v / scale);
    q = fminf(fmaxf(q, -8.f), 7.f);
    orow[c] = f32bf(q * scale);
  }
}
__global__ void k_silu_mul(const uint16_t* __restrict__ G, const uint16_t* __restrict__ U,
                           uint16_t* __restrict__ Hh, long n) {
  long i = blockIdx.x * (long)blockDim.x + threadIdx.x;
  if (i >= n) return;
  float g = bf16f(G[i]);
  float sg = g / (1.0f + expf(-g));
  uint16_t gb = f32bf(sg);
  Hh[i] = f32bf(bf16f(gb) * bf16f(U[i]));
}

// ---------------- i4l (Hadamard-rotated int4) ----------------
// Weights stored as: N rows x K/2 bytes of 4-bit two's-complement codes
// (lo nib = even col), then N x (K/256) fp16 scales. Dequant gives W'
// (the rotated weights); W[n, 256c:(c+1)256] = W'[same] @ Hadamard256/16.
// Acts are rotated by the same H before quantization (k_actq_had), so
// fp16 GEMM on (a', W') equals the engine's int4 W4A4 products exactly.
__device__ inline float fp16f(uint16_t h) {
  uint32_t sign = (uint32_t)(h & 0x8000u) << 16;
  uint32_t exp = (h >> 10) & 0x1Fu;
  uint32_t man = h & 0x3FFu;
  if (exp == 0) { float d = (float)man * (1.0f / 16777216.0f); return sign ? -d : d; }
  if (exp == 31) return man ? nanf("1") : (sign ? -INFINITY : INFINITY);
  uint32_t u = sign | ((exp - 15 + 127) << 23) | (man << 13);
  return __uint_as_float(u);
}

__global__ void k_dequant_i4l(const uint8_t* __restrict__ pay,
                              const uint16_t* __restrict__ sc,
                              uint16_t* __restrict__ out, int cols, long nr) {
  long i = blockIdx.x * (long)blockDim.x + threadIdx.x;
  long n = nr * (long)cols;
  if (i >= n) return;
  int col = (int)(i % cols);
  long row = i / cols;
  uint8_t b = pay[row * (cols >> 1) + (col >> 1)];
  int nib = (col & 1) ? (int)(b >> 4) : (int)(b & 0xF);
  int code = (nib > 7) ? nib - 16 : nib;
  float s = fp16f(sc[row * (cols >> 8) + (col >> 8)]);
  out[i] = f32bf((float)code * s);
}

// in-place Hadamard-256 along cols, per (row, 256-chunk): X = X @ H/16
__global__ void k_had_cols(uint16_t* __restrict__ X) {
  __shared__ float sh[256];
  int tid = threadIdx.x;
  long base = (long)blockIdx.x * 256;
  sh[tid] = bf16f(X[base + tid]);
  __syncthreads();
  for (int len = 1; len < 256; len <<= 1) {
    float a = sh[tid], b = sh[tid ^ len];
    __syncthreads();
    sh[tid] = ((tid & len) ? b - a : a + b);
    __syncthreads();
  }
  X[base + tid] = f32bf(sh[tid] * (1.0f / 16.0f));
}

// W4A4 exact act-quant emission (engine k_actq<true> semantics, verified
// bit-exact against the original): H-rotate the row's 256-chunk, scale =
// fp16(absmax/7), q = rint(v/scale) clamped [-7,7], emit packed codes (lo nib
// = even col) + one fp16 scale per (row, 256-chunk). Input NOT modified.
__device__ inline uint16_t f32h(float x) {
  uint32_t u = __float_as_uint(x);
  uint32_t sign = (u >> 16) & 0x8000u;
  int32_t e = (int32_t)((u >> 23) & 0xFFu) - 127 + 15;
  uint32_t m = u & 0x7FFFFFu;
  if (((u >> 23) & 0xFFu) == 255) return (uint16_t)(sign | 0x7C00u | (m ? 0x200u : 0));
  if (e >= 31) return (uint16_t)(sign | 0x7C00u);
  if (e <= 0) {
    if (e < -10) return (uint16_t)sign;
    m |= 0x800000u;
    int sh = 14 - e;
    uint32_t r = m >> sh, rem = m & ((1u << sh) - 1u), half = 1u << (sh - 1);
    r += (rem > half || (rem == half && (r & 1))) ? 1u : 0u;
    return (uint16_t)(sign | r);
  }
  uint32_t r = ((uint32_t)e << 10) | (m >> 13);
  uint32_t rem = m & 0x1FFFu;
  r += (rem > 0x1000u || (rem == 0x1000u && (r & 1))) ? 1u : 0u;
  return (uint16_t)(sign | r);
}

__global__ void k_actq_emit(const uint16_t* __restrict__ X, uint8_t* __restrict__ C,
                            uint16_t* __restrict__ S, long cols) {
  __shared__ float sh[256], sm[256];
  __shared__ uint8_t sc[256];
  int tid = threadIdx.x;
  long tok = blockIdx.x;
  int chunk = blockIdx.y;
  long base = tok * cols + (long)chunk * 256;
  sh[tid] = bf16f(X[base + tid]);
  __syncthreads();
  for (int len = 1; len < 256; len <<= 1) {
    float a = sh[tid], b = sh[tid ^ len];
    __syncthreads();
    sh[tid] = ((tid & len) ? b - a : a + b);
    __syncthreads();
  }
  float v = sh[tid] * (1.0f / 16.0f);
  sm[tid] = fabsf(v);
  __syncthreads();
  for (int off = 128; off > 0; off >>= 1) {
    if (tid < off) sm[tid] = fmaxf(sm[tid], sm[tid + off]);
    __syncthreads();
  }
  uint16_t sb = f32h(sm[0] / 7.0f);
  float scale = fp16f(sb);
  if (tid == 0) S[tok * (cols >> 8) + chunk] = sb;
  float q = 0.f;
  if (scale > 0.f) {                    // engine: exec-masked when scale == 0
    q = rintf(v / scale);
    q = fminf(fmaxf(q, -7.f), 7.f);
  }
  sc[tid] = (uint8_t)((int)q & 0xF);
  __syncthreads();
  if (tid < 128) C[tok * (cols >> 1) + chunk * 128 + tid] =
      (uint8_t)(sc[2 * tid] | (sc[2 * tid + 1] << 4));
}

// Engine k_gemm_i4 semantics (verified vs dumped outputs): per 256-chunk exact
// int dot of the 4-bit codes, scaled by f32(act_scale * weight_scale), f32
// accumulated over chunks in ascending order, one bf16 round at the end.
__global__ void k_gemm_i4emul(const uint8_t* __restrict__ ac, const uint16_t* __restrict__ as,
                              const uint8_t* __restrict__ wc, const uint16_t* __restrict__ ws,
                              uint16_t* __restrict__ C, int N, long K) {
  int m = blockIdx.x;
  int n = blockIdx.y * blockDim.x + threadIdx.x;
  if (n >= N) return;
  long per = K >> 1;
  int nsc = (int)(K >> 8);
  const uint8_t* arow = ac + (size_t)m * per;
  const uint8_t* wrow = wc + (size_t)n * per;
  const uint16_t* asr = as + (size_t)m * nsc;
  const uint16_t* wsr = ws + (size_t)n * nsc;
  __shared__ uint8_t sab[128];
  float acc = 0.f;
  for (int c = 0; c < nsc; c++) {
    if (threadIdx.x < 128) sab[threadIdx.x] = arow[c * 128 + threadIdx.x];
    __syncthreads();
    const uint8_t* wp = wrow + c * 128;
    int dot = 0;
    #pragma unroll 8
    for (int j = 0; j < 128; j++) {
      uint8_t ab = sab[j], wb = wp[j];
      dot += ((ab & 15) - ((ab & 8) << 1)) * ((wb & 15) - ((wb & 8) << 1))
           + ((ab >> 4) - ((ab & 0x80) >> 3)) * ((wb >> 4) - ((wb & 0x80) >> 3));
    }
    acc += (float)dot * (fp16f(asr[c]) * fp16f(wsr[c]));
    __syncthreads();
  }
  C[(size_t)m * N + n] = f32bf(acc);
}

// W4A4 exact act-quant emulation: H-rotate the row's 256-chunk, quantize
// with chunk scale absmax/7, rotate back. X ends as H(quant(H x)).
__global__ void k_actq_had(uint16_t* __restrict__ X) {
  __shared__ float sh[256], sm[256];
  int tid = threadIdx.x;
  long base = (long)blockIdx.x * 256;
  sh[tid] = bf16f(X[base + tid]);
  __syncthreads();
  for (int len = 1; len < 256; len <<= 1) {
    float a = sh[tid], b = sh[tid ^ len];
    __syncthreads();
    sh[tid] = ((tid & len) ? b - a : a + b);
    __syncthreads();
  }
  float v = sh[tid] * (1.0f / 16.0f);
  sm[tid] = fabsf(v);
  __syncthreads();
  for (int off = 128; off > 0; off >>= 1) {
    if (tid < off) sm[tid] = fmaxf(sm[tid], sm[tid + off]);
    __syncthreads();
  }
  float scale = bf16f(f32bf(sm[0] / 7.0f));
  if (scale <= 0.f) scale = 1.0f;
  float q = rintf(v / scale);
  q = fminf(fmaxf(q, -8.f), 7.f);
  sh[tid] = q * scale;
  __syncthreads();
  for (int len = 1; len < 256; len <<= 1) {
    float a = sh[tid], b = sh[tid ^ len];
    __syncthreads();
    sh[tid] = ((tid & len) ? b - a : a + b);
    __syncthreads();
  }
  X[base + tid] = f32bf(sh[tid] * (1.0f / 16.0f));
}

// ---------------- logits reduction ----------------
__global__ void k_bf16logits(float* Lg, long n) {
  long i = blockIdx.x * (long)blockDim.x + threadIdx.x;
  if (i >= n) return;
  Lg[i] = bf16f(f32bf(Lg[i]));
}
__global__ void k_topnll(const float* __restrict__ Lg, const int* __restrict__ rowmap,
                         const int* __restrict__ tgt, int* __restrict__ pred, float* __restrict__ nll,
                         int V, long T) {
  long row = blockIdx.x; if (row >= T) return;
  __shared__ float smx[256]; __shared__ int smi[256]; __shared__ float ss[256];
  int tid = threadIdx.x;
  const float* L = Lg + (long)rowmap[row] * V;
  float mx = -3.0e38f; int mi = 0;
  for (int v = tid; v < V; v += 256) { float x = L[v]; if (x > mx) { mx = x; mi = v; } }
  smx[tid] = mx; smi[tid] = mi;
  __syncthreads();
  for (int off = 128; off > 0; off >>= 1) {
    if (tid < off && smx[tid + off] > smx[tid]) { smx[tid] = smx[tid + off]; smi[tid] = smi[tid + off]; }
    __syncthreads();
  }
  float gmx = smx[0];
  int gmi = smi[0];
  float s = 0.f;
  for (int v = tid; v < V; v += 256) s += expf(L[v] - gmx);
  ss[tid] = s; __syncthreads();
  for (int off = 128; off > 0; off >>= 1) { if (tid < off) ss[tid] += ss[tid + off]; __syncthreads(); }
  if (tid == 0) {
    pred[row] = gmi;
    nll[row] = (gmx + logf(ss[0])) - L[tgt[row]];
  }
}
__global__ void k_argmax1(const float* __restrict__ L, int* __restrict__ pred, int V) {
  __shared__ float smx[256]; __shared__ int smi[256];
  int tid = threadIdx.x;
  float mx = -3.0e38f; int mi = 0;
  for (int v = tid; v < V; v += 256) { float x = L[v]; if (x > mx) { mx = x; mi = v; } }
  smx[tid] = mx; smi[tid] = mi;
  __syncthreads();
  for (int off = 128; off > 0; off >>= 1) {
    if (tid < off && smx[tid + off] > smx[tid]) { smx[tid] = smx[tid + off]; smi[tid] = smi[tid + off]; }
    __syncthreads();
  }
  if (tid == 0) pred[0] = smi[0];
}

// ================= host side =================
struct Ent { char name[96]; long off, size; int dt; };
static std::vector<Ent> g_tbl;
static FILE* g_f = nullptr;
static char nm[160];

static void load_table() {
  fseek(g_f, 0x08, SEEK_SET);
  uint64_t n; if (fread(&n, 8, 1, g_f) != 1) { printf("hdr read fail\n"); exit(1); }
  fseek(g_f, 0x68, SEEK_SET);
  g_tbl.resize(n);
  std::vector<uint8_t> raw((size_t)n * 0xA0);
  if (fread(raw.data(), 1, raw.size(), g_f) != raw.size()) { printf("table read fail\n"); exit(1); }
  for (uint64_t i = 0; i < n; i++) {
    const uint8_t* e = raw.data() + i * 0xA0;
    memcpy(g_tbl[i].name, e, 96); g_tbl[i].name[95] = 0;
    memcpy(&g_tbl[i].off, e + 0x88, 8);
    memcpy(&g_tbl[i].size, e + 0x90, 8);
    uint64_t f1; memcpy(&f1, e + 0x60, 8);
    g_tbl[i].dt = (int)(f1 & 0xFFFFFFFF);
  }
}
static const Ent* find(const char* name) {
  for (auto& e : g_tbl) if (strcmp(e.name, name) == 0) return &e;
  printf("tensor not found: %s\n", name); exit(1);
}

static uint8_t*  h_stage;    // 64 MB pinned
static uint8_t*  h_stage2;   // 1 MB pinned
static uint8_t*  d_raw;      // 64 MB
static uint16_t* d_raws;     // 512 KB
static float*    d_cb;

static void rd(long off, void* dst, size_t n) {
  fseek(g_f, off, SEEK_SET);
  if (fread(dst, 1, n, g_f) != n) { printf("short read @%ld\n", off); exit(1); }
}
static void up(void* d, const void* h, size_t n) { CK(hipMemcpy(d, h, n, hipMemcpyHostToDevice)); }

// fp8r -> device bf16, row-chunked (scales at tensor tail)
static void load_fp8r(const char* name, long rows, long cols, uint16_t* d_out) {
  const Ent* e = find(name);
  long chunk = (long)(32e6 / cols); if (chunk < 1) chunk = 1;
  for (long r0 = 0; r0 < rows; r0 += chunk) {
    long nr = chunk < rows - r0 ? chunk : rows - r0;
    rd(e->off + r0 * cols, h_stage, (size_t)nr * cols);
    up(d_raw, h_stage, (size_t)nr * cols);
    rd(e->off + rows * cols + r0 * 2, h_stage2, (size_t)nr * 2);
    up(d_raws, h_stage2, (size_t)nr * 2);
    long n = nr * cols;
    k_dequant_fp8r<<<GRID1(n), 256>>>(d_raw, d_raws, d_out + r0 * cols, cols);
    CK(hipDeviceSynchronize());   // d_raw is reused next chunk
  }
}
// q4c -> device bf16, row-chunked (64B codebook + per-row [payload|scales])
static void load_q4c(const char* name, long rows, long cols, uint16_t* d_out) {
  const Ent* e = find(name);
  rd(e->off, h_stage, 64); up(d_cb, h_stage, 64);
  long per = cols / 2 + cols / 16;
  long chunk = (long)(32e6 / per); if (chunk < 1) chunk = 1;
  for (long r0 = 0; r0 < rows; r0 += chunk) {
    long nr = chunk < rows - r0 ? chunk : rows - r0;
    rd(e->off + 64 + r0 * per, h_stage, (size_t)nr * per);
    up(d_raw, h_stage, (size_t)nr * per);
    long n = nr * cols;
    k_dequant_q4c<<<GRID1(n), 256>>>(d_cb, d_raw, d_out + r0 * cols, cols);
    CK(hipDeviceSynchronize());
  }
}
// i4l -> device bf16 (unrotated W'), row-chunked:
//   [N x K/2 code bytes][N x (K/256) fp16 scales]
static void load_i4l(const char* name, long rows, long cols, uint16_t* d_out) {
  const Ent* e = find(name);
  long per = cols >> 1;
  long nsc = cols >> 8;
  long chunk = (long)(24e6 / per); if (chunk < 1) chunk = 1;
  for (long r0 = 0; r0 < rows; r0 += chunk) {
    long nr = chunk < rows - r0 ? chunk : rows - r0;
    rd(e->off + r0 * per, h_stage, (size_t)nr * per);
    up(d_raw, h_stage, (size_t)nr * per);
    rd(e->off + rows * per + r0 * nsc * 2, h_stage2, (size_t)nr * nsc * 2);
    up(d_raws, h_stage2, (size_t)nr * nsc * 2);
    long n = nr * cols;
    k_dequant_i4l<<<GRID1(n), 256>>>(d_raw, d_raws, d_out + r0 * cols, (int)cols, nr);
    k_had_cols<<<(unsigned)(nr * (cols >> 8)), 256>>>(d_out + r0 * cols);
    CK(hipDeviceSynchronize());
  }
}
static const Ent* try_find(const char* name) {
  for (auto& e : g_tbl) if (strcmp(e.name, name) == 0) return &e;
  return nullptr;
}
static void load_bf16(const char* name, long bytes, void* d_out) {
  const Ent* e = find(name);
  long chunk = 32 << 20;
  for (long o = 0; o < bytes; o += chunk) {
    long nb = chunk < bytes - o ? chunk : bytes - o;
    rd(e->off + o, h_stage, nb);
    up((uint8_t*)d_out + o, h_stage, nb);
  }
}

static void load_auto(const char* name, long rows, long cols, uint16_t* d_out) {
  if (find(name)->dt == 5) load_q4c(name, rows, cols, d_out);
  else load_fp8r(name, rows, cols, d_out);
}

// ---------------- device weight pool ----------------
// All quantized payloads resident on device; forward() dequants per use. Falls
// back to streaming (per-use file reads) when the allocation doesn't fit.
static uint8_t* d_wpool = nullptr;
static size_t g_wpool_cap = 0, g_wpool_used = 0;
struct WEnt { size_t off; long rows, cols; int dt; };
static std::map<std::string, WEnt> g_wmap;

static bool pool_add(const char* name, long rows, long cols) {
  const Ent* e = find(name);
  if (g_wpool_used + (size_t)e->size > g_wpool_cap) return false;
  long chunk = 32 << 20;
  for (long o = 0; o < e->size; o += chunk) {
    long nb = chunk < e->size - o ? chunk : e->size - o;
    rd(e->off + o, h_stage, nb);
    up(d_wpool + g_wpool_used + o, h_stage, nb);
  }
  g_wmap[name] = {g_wpool_used, rows, cols, e->dt};
  g_wpool_used += (size_t)e->size;
  return true;
}

static int g_prefill = 0;  // 1 = multi-token forward: use Hadamard-rotated i4l weights

static int gate_dt5(int l) {
  char nm[176];
  snprintf(nm, sizeof(nm), "layers.%d.mlp.gate_proj.weight", l);
  const Ent* e = find(nm);
  return e && e->dt == 5;
}

static void getw(const char* name, long rows, long cols, uint16_t* d_out) {
  auto it = g_wmap.find(name);
  if (it != g_wmap.end()) {
    const WEnt& w = it->second;
    long n = rows * cols;
    if (w.dt == 5)
      k_dequant_q4c<<<GRID1(n), 256>>>((const float*)(d_wpool + w.off),
                                       d_wpool + w.off + 64, d_out, cols);
    else
      k_dequant_fp8r<<<GRID1(n), 256>>>(d_wpool + w.off,
                                        (const uint16_t*)(d_wpool + w.off + (size_t)rows * cols),
                                        d_out, cols);
    return;
  }
  if (g_prefill && find(name)->dt == 5) {
    // Only q4c-based tensors take the rotated-i4l W4A4 route (engine rule:
    // base dt 5 -> i4l W4A4; base dt 6 (fp8r) -> dequant + bf16 GEMM).
    char nm2[176];
    snprintf(nm2, sizeof(nm2), "%s.i4l", name);
    if (try_find(nm2)) { load_i4l(nm2, rows, cols, d_out); return; }
  }
  load_auto(name, rows, cols, d_out);
}

// stream one i4l tensor's raw codes + fp16 scales to device staging buffers
// (used by the exact W4A4 int-dot GEMM emulation; no dequant, no unrotation)
static uint8_t* d_wc[3]; static uint16_t* d_ws[3];
static void load_i4l_raw(const char* name, long rows, long cols, int slot) {
  const Ent* e = find(name);
  long per = cols >> 1;
  long nsc = cols >> 8;
  uint8_t* dc = d_wc[slot];
  uint16_t* ds = d_ws[slot];
  long chunk = (long)(24e6 / per); if (chunk < 1) chunk = 1;
  for (long r0 = 0; r0 < rows; r0 += chunk) {
    long nr = chunk < rows - r0 ? chunk : rows - r0;
    rd(e->off + r0 * per, h_stage, (size_t)nr * per);
    up(dc + r0 * per, h_stage, (size_t)nr * per);
    rd(e->off + rows * per + r0 * nsc * 2, h_stage2, (size_t)nr * nsc * 2);
    up((uint8_t*)(ds + r0 * nsc), h_stage2, (size_t)nr * nsc * 2);
  }
  CK(hipDeviceSynchronize());
}

// fused dequant GEMV (decode steps, M == 1): C[n] = sum_k A[k] * dequant(W[n][k]).
// fp8r: per-thread 16-k strips, two f32 accumulators, xor-butterfly reduce, row
// scale applied once at the end (mirrors k_gemv's structure); q4c: per-element
// codebook*groupscale. f32 accumulate throughout, one bf16 round at the output.
template<int Q4C>
__global__ void k_gemv_dq(const uint16_t* __restrict__ A,
                          const uint8_t* __restrict__ Wp,
                          const uint16_t* __restrict__ Ws,
                          const float* __restrict__ CB, long rowstride,
                          uint16_t* __restrict__ C, int K) {
  int n = blockIdx.x;
  const uint8_t* row = Wp + (size_t)n * rowstride;
  int tid = threadIdx.x;
  float acc0 = 0.f, acc1 = 0.f;
  for (int k0 = tid * 16; k0 < K; k0 += 4096) {
    int ke = k0 + 16 > K ? K : k0 + 16;
    for (int k = k0; k < ke; k += 2) {
      float a0 = bf16f(A[k]), w0, a1 = 0.f, w1 = 0.f;
      if (Q4C) {
        uint8_t b0 = row[k >> 1];
        w0 = CB[b0 & 0xF] * dequant_fp8_e4m3(row[(K >> 1) + (k >> 4)]);
        if (k + 1 < K) {
          uint8_t b1 = row[(k + 1) >> 1];
          a1 = bf16f(A[k + 1]);
          w1 = CB[(b1 >> 4) & 0xF] * dequant_fp8_e4m3(row[(K >> 1) + ((k + 1) >> 4)]);
        }
      } else {
        w0 = dequant_fp8_e4m3(row[k]);
        if (k + 1 < K) { a1 = bf16f(A[k + 1]); w1 = dequant_fp8_e4m3(row[k + 1]); }
      }
      acc0 = fmaf(a0, w0, acc0);
      if (k + 1 < K) acc1 = fmaf(a1, w1, acc1);
    }
  }
  float acc = acc0 + acc1;
  if (!Q4C) acc *= bf16f(Ws[n]);
  __shared__ float sh[256];
  sh[tid] = acc;
  __syncthreads();
  for (int off = 128; off > 0; off >>= 1) {
    if (tid < off) sh[tid] += sh[tid + off];
    __syncthreads();
  }
  if (tid == 0) C[n] = f32bf(sh[0]);
}

// plain bf16 GEMV with f32 output (lm_head at T == 1)
__global__ void k_gemv_bf16f(const uint16_t* __restrict__ A, const uint16_t* __restrict__ W,
                             float* __restrict__ C, int K) {
  int n = blockIdx.x;
  const uint16_t* row = W + (size_t)n * K;
  int tid = threadIdx.x;
  float acc0 = 0.f, acc1 = 0.f;
  for (int k0 = tid * 16; k0 < K; k0 += 4096) {
    int ke = k0 + 16 > K ? K : k0 + 16;
    for (int k = k0; k < ke; k += 2) {
      acc0 = fmaf(bf16f(A[k]), bf16f(row[k]), acc0);
      if (k + 1 < K) acc1 = fmaf(bf16f(A[k + 1]), bf16f(row[k + 1]), acc1);
    }
  }
  float acc = acc0 + acc1;
  __shared__ float sh[256];
  sh[tid] = acc;
  __syncthreads();
  for (int off = 128; off > 0; off >>= 1) {
    if (tid < off) sh[tid] += sh[tid + off];
    __syncthreads();
  }
  if (tid == 0) C[n] = sh[0];
}

static bool gemv_dq(const uint16_t* A, const char* name, uint16_t* C) {  if (!d_wpool) return false;
  auto it = g_wmap.find(name);
  if (it == g_wmap.end()) return false;
  const WEnt& w = it->second;
  if (w.dt == 5)
    k_gemv_dq<1><<<dim3((unsigned)w.rows), 256>>>(A, d_wpool + w.off + 64, nullptr,
                                                  (const float*)(d_wpool + w.off),
                                                  w.cols / 2 + w.cols / 16, C, (int)w.cols);
  else
    k_gemv_dq<0><<<dim3((unsigned)w.rows), 256>>>(A, d_wpool + w.off,
                                                  (const uint16_t*)(d_wpool + w.off + (size_t)w.rows * w.cols),
                                                  nullptr, w.cols, C, (int)w.cols);
  return true;
}

// ---------------- device memory ----------------
static uint16_t *d_embed, *d_lmhead;
static uint16_t *d_ln_in_all, *d_ln_post_all, *d_final_norm;
static uint16_t *d_gdn_conv, *d_gdn_a, *d_gdn_b, *d_gdn_norm;
static uint16_t *d_qnorm, *d_knorm;
static uint16_t *d_qkv, *d_z, *d_outp;
static uint16_t *d_q, *d_k, *d_v, *d_o;
static uint16_t *d_gate, *d_up, *d_down;
static uint8_t* d_qc; static uint16_t* d_qs;
static uint16_t *d_convstate;                 // [48][10240][3]
static uint16_t *d_kcache, *d_vcache;         // [16][CXT][4][256]
static uint16_t *d_x, *d_ln1, *d_ln2, *d_tmp1;
static uint16_t *d_qkvraw, *d_qkvc, *d_ab_a, *d_ab_b, *d_zout, *d_co, *d_gn;
static uint16_t *d_qg, *d_kk, *d_vv, *d_att, *d_qn, *d_kn;
static uint16_t *d_g17, *d_u17, *d_act;
static float *d_scores, *d_logits;
static uint16_t *d_pbf;
static int *d_tok, *d_tgt, *d_pred;
static float *d_nll;

// host
static float* h_S;            // [48][48][128][128] GDN state f32 (GDN-ordinal layer index)
static float* h_alog;         // [48][48]
static float* h_dtbias;       // [48][48]
static uint16_t* h_qkvbf;     // [TMAX][10240]
static uint16_t* h_abbf;      // [TMAX][48]
static float* h_af;           // [TMAX][48]
static float* h_bf;           // [TMAX][48]
static uint16_t* h_co;        // [TMAX][6144]
static int* h_prompt; static int h_nprompt;
static float h_qscale;

static inline float softplusf(float x) { return x > 20.0f ? x : log1pf(expf(x)); }
static inline int gdn_ord(int l) { return l - (l + 1) / 4; }   // layer idx -> GDN ordinal (0..47)
static inline int fa_ord(int l)  { return l / 4; }             // layer idx -> FA ordinal (0..15)

static int g_dbg = 0;
static int g_steps = 0;
static int g_bf16logits = 0;
static int g_actq = 0;    // engine-default serve prefill is CLEAN; HALO_ACTQ=3 emulates
                          // the forced-bench W4A4 prefill (mean-NLL 5.9406 vs 5.918531)
static int g_dnprep = 1;  // 1: round l2normed q/k to bf16 (engine inter-kernel dtype)
static int g_beta32 = 0;  // 0: beta stays f32 (k_dn_gates emits f32) — matches engine clean path
static void dbg_all(const char* tag, const uint16_t* d_buf, long rows, long width) {
  static uint16_t tmp[TMAX * 17408];
  CK(hipMemcpy(tmp, d_buf, rows * width * 2, hipMemcpyDeviceToHost));
  double mx = 0, sm = 0; long nan = 0;
  for (long i = 0; i < rows * width; i++) {
    float v = h_bf16f(tmp[i]);
    if (v != v) nan++;
    if (fabs(v) > mx) mx = fabs(v);
    sm += v;
  }
  printf("  [%s] allrows mean=%.4g maxabs=%.4g nan=%ld\n", tag, sm / (rows * width), mx, nan);
}
static void dbg_row(const char* tag, const uint16_t* d_buf, long row, long width) {
  static uint16_t tmp[5120];
  CK(hipMemcpy(tmp, d_buf + row * width, width * 2, hipMemcpyDeviceToHost));
  double mx = 0, sm = 0; int nan = 0;
  for (long i = 0; i < width; i++) {
    float v = h_bf16f(tmp[i]);
    if (v != v) nan++;
    if (fabs(v) > mx) mx = fabs(v);
    sm += v;
  }
  printf("  [%s] mean=%.4g maxabs=%.4g nan=%d v0..3= %.4g %.4g %.4g %.4g\n",
         tag, sm / width, mx, nan, h_bf16f(tmp[0]), h_bf16f(tmp[1]), h_bf16f(tmp[2]), h_bf16f(tmp[3]));
}
static void dbg_f32(const char* tag, const float* d_buf, long row, long width) {
  static float tmp[248320];
  CK(hipMemcpy(tmp, d_buf + row * width, width * 4, hipMemcpyDeviceToHost));
  double mx = 0, sm = 0; int nan = 0;
  for (long i = 0; i < width; i++) {
    float v = tmp[i];
    if (v != v) nan++;
    if (fabs(v) > mx) mx = fabs(v);
    sm += v;
  }
  printf("  [%s] mean=%.4g maxabs=%.4g nan=%d v0..3= %.4g %.4g %.4g %.4g\n",
         tag, sm / width, mx, nan, tmp[0], tmp[1], tmp[2], tmp[3]);
}

// GDN: fla-style CHUNKED gated delta rule (f32, host), chunk 64.
// Spec: torch_chunk_gated_delta_rule in qwen35_ref.py (the engine runs k_dnc_ut/scan
// kernels — the chunked algorithm; the sequential form is numerically different).
static void gdn_scan(int T, const uint16_t* qkvbf, const float* af, const float* bfv,
                     int ord, uint16_t* out) {
  const float* al = h_alog + (long)ord * HV;
  const float* dt = h_dtbias + (long)ord * HV;
  const int CS = 64;
  const int nch = (T + CS - 1) / CS;
  const int NP = nch * CS;
#ifdef _OPENMP
  #pragma omp parallel for schedule(static)
#endif
  for (int hh = 0; hh < HV; hh++) {
    float* S = h_S + ((long)ord * HV + hh) * DK * DV;
    std::vector<float> Q((long)NP * DK, 0.f), K((long)NP * DK, 0.f), V((long)NP * DV, 0.f);
    std::vector<float> G(NP, 0.f), B(NP, 0.f);
    for (int t = 0; t < T; t++) {
      const uint16_t* qp = qkvbf + (long)t * QKVW + (hh / 3) * DK;
      const uint16_t* kp = qkvbf + (long)t * QKVW + HK * DK + (hh / 3) * DK;
      const uint16_t* vp = qkvbf + (long)t * QKVW + HK * DK * 2 + hh * DV;
      float qs = 0.f, ks = 0.f;
      for (int i = 0; i < DK; i++) {
        float a = h_bf16f(qp[i]); Q[(long)t * DK + i] = a; qs += a * a;
        float b = h_bf16f(kp[i]); K[(long)t * DK + i] = b; ks += b * b;
      }
      float qi = 1.0f / sqrtf(qs + EPS) * h_qscale, ki = 1.0f / sqrtf(ks + EPS);
      for (int i = 0; i < DK; i++) {
        // engine passes l2normed q/k between kernels in bf16 (k_dnc_prep -> k_dnc_scan)
        float qv = Q[(long)t * DK + i] * qi, kv_ = K[(long)t * DK + i] * ki;
        if (g_dnprep) { qv = h_bf16f(h_f32bf(qv)); kv_ = h_bf16f(h_f32bf(kv_)); }
        Q[(long)t * DK + i] = qv; K[(long)t * DK + i] = kv_;
      }
      for (int i = 0; i < DV; i++) V[(long)t * DV + i] = h_bf16f(vp[i]);
      G[t] = -expf(al[hh]) * softplusf(af[(long)t * HV + hh] + dt[hh]);
      B[t] = 1.0f / (1.0f + expf(-bfv[(long)t * HV + hh]));   // k_dn_gates emits f32 beta
      if (!g_beta32) B[t] = h_bf16f(h_f32bf(B[t]));
    }
    std::vector<float> vb((long)NP * DV), kb((long)NP * DK);
    for (int t = 0; t < NP; t++) {
      for (int i = 0; i < DV; i++) vb[(long)t * DV + i] = V[(long)t * DV + i] * B[t];
      for (int i = 0; i < DK; i++) kb[(long)t * DK + i] = K[(long)t * DK + i] * B[t];
    }
    std::vector<float> pd((long)CS * CS), ut((long)CS * CS), intra((long)CS * CS);
    std::vector<float> nv((long)CS * DV), kcd((long)CS * DK), qd((long)CS * DK), kd((long)CS * DK);
    std::vector<float> vn((long)CS * DV), inter((long)CS * DV), cum(CS);
    for (int c = 0; c < nch; c++) {
      int t0 = c * CS;
      { float acc = 0.f;
        for (int a = 0; a < CS; a++) { acc += G[t0 + a]; cum[a] = acc; } }
      for (int a = 0; a < CS; a++) {
        for (int b = 0; b < CS; b++) {
          float p = b <= a ? expf(cum[a] - cum[b]) : 0.f;
          pd[(long)a * CS + b] = p;
          float kbd = 0.f, qkd = 0.f;
          if (p != 0.f) {
            const float* kba = &kb[(long)(t0 + a) * DK];
            const float* kb2 = &K[(long)(t0 + b) * DK];
            const float* qa = &Q[(long)(t0 + a) * DK];
            for (int i = 0; i < DK; i++) { kbd += kba[i] * kb2[i]; qkd += qa[i] * kb2[i]; }
          }
          ut[(long)a * CS + b] = kbd * p;
          intra[(long)a * CS + b] = qkd * p;
        }
      }
      // unitriangular forward substitution: x[a] = b[a] - sum_{s<a} ut[a][s] x[s]
      for (int a = 0; a < CS; a++) {
        for (int j = 0; j < DV; j++) {
          float s = vb[(long)(t0 + a) * DV + j];
          for (int sc = 0; sc < a; sc++) s -= ut[(long)a * CS + sc] * nv[(long)sc * DV + j];
          nv[(long)a * DV + j] = s;
        }
        const float* kba = &kb[(long)(t0 + a) * DK];
        float ec = expf(cum[a]);
        for (int i = 0; i < DK; i++) { kcd[(long)a * DK + i] = kba[i] * ec; qd[(long)a * DK + i] = Q[(long)(t0 + a) * DK + i] * ec; }
        float ecd = expf(cum[CS - 1] - cum[a]);
        for (int i = 0; i < DK; i++) kd[(long)a * DK + i] = K[(long)(t0 + a) * DK + i] * ecd;
      }
      for (int a = 1; a < CS; a++) {
        for (int sc = 0; sc < a; sc++) {
          float u = ut[(long)a * CS + sc];
          const float* kcs = &kcd[(long)sc * DK];
          float* kca = &kcd[(long)a * DK];
          for (int i = 0; i < DK; i++) kca[i] -= u * kcs[i];
        }
      }
      const float cdec = expf(cum[CS - 1]);
      for (int a = 0; a < CS; a++) {
        const float* kca = &kcd[(long)a * DK];
        const float* qda = &qd[(long)a * DK];
        for (int j = 0; j < DV; j++) {
          float s1 = 0.f, s2 = 0.f;
          for (int i = 0; i < DK; i++) { s1 += kca[i] * S[(long)i * DV + j]; s2 += qda[i] * S[(long)i * DV + j]; }
          vn[(long)a * DV + j] = nv[(long)a * DV + j] - s1;
          inter[(long)a * DV + j] = s2;
        }
        for (int b = 0; b <= a; b++) {
          float w = intra[(long)a * CS + b];
          if (w == 0.f) continue;
          const float* vb2 = &vn[(long)b * DV];
          for (int j = 0; j < DV; j++) inter[(long)a * DV + j] += w * vb2[j];
        }
      }
      for (int a = 0; a < CS; a++)
        for (int j = 0; j < DV; j++) {
          int t = t0 + a;
          if (t < T) out[(long)t * (HV * DV) + hh * DV + j] = h_f32bf(inter[(long)a * DV + j]);
        }
      for (int i = 0; i < DK; i++) {
        float* Sp = S + (long)i * DV;
        for (int j = 0; j < DV; j++) Sp[j] *= cdec;
      }
      for (int a = 0; a < CS; a++) {
        const float* kda = &kd[(long)a * DK];
        const float* va = &vn[(long)a * DV];
        for (int i = 0; i < DK; i++) {
          float kk_ = kda[i];
          float* Sp = S + (long)i * DV;
          for (int j = 0; j < DV; j++) Sp[j] += kk_ * va[j];
        }
      }
    }
  }
}

// ---------------- forward ----------------
// T tokens at positions pos_base..; logits [T][VOC] f32 in d_logits when want_logits
static void forward(const int* d_tokens, int T, int pos_base, int want_logits) {
  g_prefill = (T > 1);
  k_embed<<<GRID1((long)T * HID), 256>>>(d_tokens, d_embed, d_x, HID);
  CK(hipDeviceSynchronize());
  if (g_dbg) { printf("after embed:\n"); dbg_row("embed", d_x, 0, HID); }
  for (int l = 0; l < NL; l++) {
    double tw = now_s();
    int fa = (l % 4 == 3);
    int gi = fa ? fa_ord(l) : gdn_ord(l);
    bool fused = (T == 1 && d_wpool);   // decode step: fused dequant-GEMV, no dequanted W
    int w4 = g_actq && T >= 64 && d_wc[0] && gate_dt5(l);
    // ---- load weights for this layer (or fused from the pool) ----
    if (fa) {
      if (!fused) {
        snprintf(nm, sizeof(nm), "layers.%d.self_attn.q_proj.weight", l);  getw(nm, QH * HD * 2, HID, d_q);
        snprintf(nm, sizeof(nm), "layers.%d.self_attn.k_proj.weight", l);  getw(nm, KVH * HD, HID, d_k);
        snprintf(nm, sizeof(nm), "layers.%d.self_attn.v_proj.weight", l);  getw(nm, KVH * HD, HID, d_v);
        snprintf(nm, sizeof(nm), "layers.%d.self_attn.o_proj.weight", l);  getw(nm, HID, QH * HD, d_o);
      }
    } else {
      if (!fused) {
        snprintf(nm, sizeof(nm), "layers.%d.linear_attn.in_proj_qkv.weight", l); getw(nm, QKVW, HID, d_qkv);
        snprintf(nm, sizeof(nm), "layers.%d.linear_attn.in_proj_z.weight", l);   getw(nm, HV * DV, HID, d_z);
        snprintf(nm, sizeof(nm), "layers.%d.linear_attn.out_proj.weight", l);    getw(nm, HID, HV * DV, d_outp);
      }
    }
    if (!fused) {
      if (w4) {
        snprintf(nm, sizeof(nm), "layers.%d.mlp.gate_proj.weight.i4l", l);
        load_i4l_raw(nm, INTM, HID, 0);
        snprintf(nm, sizeof(nm), "layers.%d.mlp.up_proj.weight.i4l", l);
        load_i4l_raw(nm, INTM, HID, 1);
        snprintf(nm, sizeof(nm), "layers.%d.mlp.down_proj.weight.i4l", l);
        load_i4l_raw(nm, HID, INTM, 2);
      } else {
        snprintf(nm, sizeof(nm), "layers.%d.mlp.gate_proj.weight", l); getw(nm, INTM, HID, d_gate);
        snprintf(nm, sizeof(nm), "layers.%d.mlp.up_proj.weight", l);   getw(nm, INTM, HID, d_up);
        snprintf(nm, sizeof(nm), "layers.%d.mlp.down_proj.weight", l); getw(nm, HID, INTM, d_down);
      }
      CK(hipDeviceSynchronize());
    }
    double tload = now_s() - tw;

    // ---- input layernorm ----
    dumpx(d_x);
    k_rmsnorm<<<T, 256>>>(d_x, d_ln_in_all + (long)l * HID, d_ln1, HID, T, 1);
    CK(hipDeviceSynchronize());

    double tc = now_s();
    if (!fa) {
      // ---- GDN branch ----
      if (fused) {
        snprintf(nm, sizeof(nm), "layers.%d.linear_attn.in_proj_qkv.weight", l); gemv_dq(d_ln1, nm, d_qkvraw);
        snprintf(nm, sizeof(nm), "layers.%d.linear_attn.in_proj_z.weight", l);   gemv_dq(d_ln1, nm, d_zout);
      } else {
        gemm(d_ln1, d_qkv, d_qkvraw, T, QKVW, HID);
        gemm(d_ln1, d_z, d_zout, T, HV * DV, HID);
      }
      if (T > 1) {
        k_conv1d<<<GRID1((long)T * QKVW), 256>>>(d_qkvraw, d_gdn_conv + (long)gi * QKVW * 4, d_qkvc, T);
        k_conv_setstate<<<QKVW / 256, 256>>>(d_qkvraw, d_convstate + (long)gi * QKVW * 3, T);
      } else {
        k_conv1d_step<<<QKVW / 256, 256>>>(d_qkvraw, d_gdn_conv + (long)gi * QKVW * 4, d_qkvc,
                                           d_convstate + (long)gi * QKVW * 3);
      }
      gemm(d_ln1, d_gdn_a + (long)gi * HV * HID, d_ab_a, T, HV, HID);
      gemm(d_ln1, d_gdn_b + (long)gi * HV * HID, d_ab_b, T, HV, HID);
      CK(hipDeviceSynchronize());
      CK(hipMemcpy(h_qkvbf, d_qkvc, (long)T * QKVW * 2, hipMemcpyDeviceToHost));
      CK(hipMemcpy(h_abbf, d_ab_a, (long)T * HV * 2, hipMemcpyDeviceToHost));
      for (long i = 0; i < (long)T * HV; i++) h_af[i] = h_bf16f(h_abbf[i]);
      CK(hipMemcpy(h_abbf, d_ab_b, (long)T * HV * 2, hipMemcpyDeviceToHost));
      for (long i = 0; i < (long)T * HV; i++) h_bf[i] = h_bf16f(h_abbf[i]);
      double ts = now_s();
      gdn_scan(T, h_qkvbf, h_af, h_bf, gi, h_co);
      double tscan = now_s() - ts;
      if (g_dbg) {
        double qm = 0, cm = 0, am = 0, bm = 0;
        for (long i = 0; i < (long)T * QKVW; i++) {
          float v = h_bf16f(h_qkvbf[i]);
          if (fabs(v) > qm) qm = fabs(v);
        }
        for (long i = 0; i < (long)T * HV * DV; i++) {
          float v = h_bf16f(h_co[i]);
          if (fabs(v) > cm) cm = fabs(v);
        }
        for (long i = 0; i < (long)T * HV; i++) {
          am = fmax(am, fabs(h_af[i]));
          bm = fmax(bm, fabs(h_bf[i]));
        }
        printf("  L%d gdn: qkvconv_max=%.3g co_max=%.3g a_max=%.3g b_max=%.3g g0=%.4g beta0=%.4g\n",
               l, qm, cm, am, bm,
               -expf(h_alog[(long)gdn_ord(l) * HV]) * softplusf(h_af[0] + h_dtbias[(long)gdn_ord(l) * HV]),
               1.f / (1.f + expf(-h_bf[0])));
      }
      up(d_co, h_co, (long)T * HV * DV * 2);
      k_gdn_gatednorm<<<GRID1((long)T * HV), 256>>>(d_co, d_zout, d_gdn_norm + (long)gi * DV, d_gn, T);
      if (fused) {
        snprintf(nm, sizeof(nm), "layers.%d.linear_attn.out_proj.weight", l); gemv_dq(d_gn, nm, d_tmp1);
      } else {
        gemm(d_gn, d_outp, d_tmp1, T, HID, HV * DV);
      }
      if (g_dbg) { dbg_all("zout", d_zout, T, HV * DV); dbg_all("gn", d_gn, T, HV * DV); dbg_all("gdn_attnout", d_tmp1, T, HID); }
      if (getenv("HALO_TIME")) printf("  L%d gdn load %.2fs scan %.3fs\n", l, tload, tscan);
    } else {
      // ---- FA branch ----
      if (fused) {
        snprintf(nm, sizeof(nm), "layers.%d.self_attn.q_proj.weight", l);  gemv_dq(d_ln1, nm, d_qg);
        snprintf(nm, sizeof(nm), "layers.%d.self_attn.k_proj.weight", l);  gemv_dq(d_ln1, nm, d_kk);
        snprintf(nm, sizeof(nm), "layers.%d.self_attn.v_proj.weight", l);  gemv_dq(d_ln1, nm, d_vv);
      } else {
        gemm(d_ln1, d_q, d_qg, T, QH * HD * 2, HID);
        gemm(d_ln1, d_k, d_kk, T, KVH * HD, HID);
        gemm(d_ln1, d_v, d_vv, T, KVH * HD, HID);
      }
      k_headnorm<<<GRID1((long)T * QH), 256>>>(d_qg, d_qnorm + (long)gi * HD, d_qn, T, QH, QH * HD * 2, HD * 2, HD);
      k_headnorm<<<GRID1((long)T * KVH), 256>>>(d_kk, d_knorm + (long)gi * HD, d_kn, T, KVH, KVH * HD, HD, HD);
      k_rope<<<GRID1((long)T * QH * NPAIR), 256>>>(d_qn, T, QH, pos_base, ROPE_MODE);
      k_rope<<<GRID1((long)T * KVH * NPAIR), 256>>>(d_kn, T, KVH, pos_base, ROPE_MODE);
      uint16_t* kc = d_kcache + (long)gi * CXT * KVH * HD;
      uint16_t* vc = d_vcache + (long)gi * CXT * KVH * HD;
      k_kvwrite<<<GRID1((long)T * KVH * HD), 256>>>(d_kn, kc, T, pos_base);
      k_kvwrite<<<GRID1((long)T * KVH * HD), 256>>>(d_vv, vc, T, pos_base);
      if (T > 1) {
        k_attn_scores<<<GRID1((long)QH * T * T), 256>>>(d_qn, kc, d_scores, T, CXT);
        k_attn_softmax<<<GRID1((long)QH * T), 256>>>(d_scores, d_pbf, T, CXT);
        k_attn_pv<<<GRID1((long)T * QH * HD), 256>>>(d_pbf, vc, d_att, T, CXT);
      } else {
        k_attn_step<<<QH, 256>>>(d_qn, kc, vc, d_att, pos_base);
      }
      k_fa_gate<<<GRID1((long)T * QH * HD), 256>>>(d_att, d_qg, T);
      if (fused) {
        snprintf(nm, sizeof(nm), "layers.%d.self_attn.o_proj.weight", l); gemv_dq(d_att, nm, d_tmp1);
      } else {
        gemm(d_att, d_o, d_tmp1, T, HID, QH * HD);
      }
      if (g_dbg) dbg_all("q_rope", d_qn, T, QH * HD);
    }
    k_add<<<GRID1((long)T * HID), 256>>>(d_x, d_tmp1, d_x, (long)T * HID);

    // ---- MLP ----
    dumpx(d_x);
    k_rmsnorm<<<T, 256>>>(d_x, d_ln_post_all + (long)l * HID, d_ln2, HID, T, 1);
    if (g_dbg) dbg_all("ln2", d_ln2, T, HID);
    { const char* dp = getenv("HALO_DUMP_LN2");
      if (dp && l == 0) { void* hb = malloc(TMAX * HID * 2);
        CK(hipMemcpy(hb, d_ln2, 87L * HID * 2, hipMemcpyDeviceToHost));
        FILE* f = fopen(dp, "wb"); fwrite(hb, 1, 87L * HID * 2, f); fclose(f); free(hb); } }
    if (w4) {
      k_actq_emit<<<dim3(T, HID / 256), 256>>>(d_ln2, d_qc, d_qs, HID);
      k_gemm_i4emul<<<dim3(T, INTM / 256), 256>>>(d_qc, d_qs, d_wc[0], d_ws[0], d_g17, INTM, HID);
      k_gemm_i4emul<<<dim3(T, INTM / 256), 256>>>(d_qc, d_qs, d_wc[1], d_ws[1], d_u17, INTM, HID);
      { const char* dp = getenv("HALO_DUMP_W4");
        if (dp && l == 0) { CK(hipDeviceSynchronize());
          void* hb = malloc(34816 * 2); void* hb2 = malloc(4096);
          CK(hipMemcpy(hb, d_g17, 34816 * 2, hipMemcpyDeviceToHost));
          CK(hipMemcpy(hb2, d_qc, 4096, hipMemcpyDeviceToHost));
          FILE* f = fopen(dp, "wb"); fwrite(hb, 1, 34816 * 2, f);
          fwrite(hb2, 1, 4096, f); fclose(f); free(hb); free(hb2);
          CK(hipMemcpy(hb2, d_qs, 4096, hipMemcpyDeviceToHost));
          f = fopen("/tmp/ours_qs.bin", "wb"); fwrite(hb2, 1, 4096, f); fclose(f); free(hb2); } }
    } else if (fused) {
      snprintf(nm, sizeof(nm), "layers.%d.mlp.gate_proj.weight", l); gemv_dq(d_ln2, nm, d_g17);
      snprintf(nm, sizeof(nm), "layers.%d.mlp.up_proj.weight", l);   gemv_dq(d_ln2, nm, d_u17);
    } else {
      gemm(d_ln2, d_gate, d_g17, T, INTM, HID);
      gemm(d_ln2, d_up, d_u17, T, INTM, HID);
    }
    if (g_dbg) { dbg_all("g17", d_g17, T, INTM); dbg_all("u17", d_u17, T, INTM); }
    k_silu_mul<<<GRID1((long)T * INTM), 256>>>(d_g17, d_u17, d_act, (long)T * INTM);
    if (w4) {
      k_actq_emit<<<dim3(T, INTM / 256), 256>>>(d_act, d_qc, d_qs, INTM);
      k_gemm_i4emul<<<dim3(T, HID / 256), 256>>>(d_qc, d_qs, d_wc[2], d_ws[2], d_tmp1, HID, INTM);
    } else if (fused) {
      snprintf(nm, sizeof(nm), "layers.%d.mlp.down_proj.weight", l); gemv_dq(d_act, nm, d_tmp1);
    } else {
      gemm(d_act, d_down, d_tmp1, T, HID, INTM);
    }
    if (g_dbg) { dbg_all("mlp_act", d_act, T, INTM); dbg_all("mlp_out", d_tmp1, T, HID); }
    k_add<<<GRID1((long)T * HID), 256>>>(d_x, d_tmp1, d_x, (long)T * HID);
    CK(hipDeviceSynchronize());
    if (g_dbg) { printf("after L%d (%s):\n", l, fa ? "fa" : "gdn"); dbg_all("x", d_x, T, HID); }
    if (getenv("HALO_TIME")) printf("  L%d load %.2fs compute %.2fs\n", l, tload, now_s() - tc);
  }
  if (want_logits) {
    dumpx(d_x);
    k_rmsnorm<<<T, 256>>>(d_x, d_final_norm, d_ln1, HID, T, 1);
    if (T == 1 && d_wpool) k_gemv_bf16f<<<VOC, 256>>>(d_ln1, d_lmhead, d_logits, HID);
    else gemm_f32(d_ln1, d_lmhead, d_logits, T, VOC, HID);
    if (g_bf16logits) k_bf16logits<<<GRID1((long)T * VOC), 256>>>(d_logits, (long)T * VOC);
    CK(hipDeviceSynchronize());
    if (g_dbg) { printf("final:\n"); dbg_row("finalnorm", d_ln1, 0, HID); dbg_f32("logits", d_logits, 0, VOC); }
  } else {
    CK(hipDeviceSynchronize());
  }
}

// ================= engine API (used by the server and the harness) =================
// Numerics validated against the original engine (see docs/halogen/TRUNK-NOTES.md):
// clean serve path (HALO_ACTQ=0): greedy stream 271 51 1618 579 1558 (5 exact), 15/24
// tokens match, residual = MFMA accumulation-order noise; forced-bench W4A4 path
// emulated by HALO_ACTQ=3 (mean-NLL 5.9406 vs 5.918531).

static int g_ready = 0;

static void trunk_reset_states() {
  CK(hipMemset(d_convstate, 0, (long)48 * QKVW * 3 * 2));
  CK(hipMemset(d_kcache, 0, (long)16 * CXT * KVH * HD * 2));
  CK(hipMemset(d_vcache, 0, (long)16 * CXT * KVH * HD * 2));
  memset(h_S, 0, (size_t)48 * HV * DK * DV * 4);
}

extern "C" int chlorine_trunk_init(const char* ckpt_path) {
  if (g_ready) return 0;
  if (getenv("HALO_ROPE")) ROPE_MODE = atoi(getenv("HALO_ROPE"));
  if (getenv("HALO_DEBUG")) g_dbg = 1;
  if (getenv("HALO_BF16LOG")) g_bf16logits = atoi(getenv("HALO_BF16LOG"));
  if (getenv("HALO_ACTQ")) g_actq = atoi(getenv("HALO_ACTQ"));
  if (getenv("HALO_DNPREP")) g_dnprep = atoi(getenv("HALO_DNPREP"));
  if (getenv("HALO_BETA32")) g_beta32 = atoi(getenv("HALO_BETA32"));

  int ndev; CK(hipGetDeviceCount(&ndev));
  if (ndev < 1) { fprintf(stderr, "trunk: no HIP device\n"); return -1; }
  CK(hipSetDevice(0));

  double t0 = now_s();
  g_f = fopen(ckpt_path, "rb");
  if (!g_f) { fprintf(stderr, "trunk: cannot open %s\n", ckpt_path); return -1; }
  load_table();
  fprintf(stderr, "trunk: table %zu tensors\n", g_tbl.size());

  CK(hipHostMalloc(&h_stage, 64 << 20));
  CK(hipHostMalloc(&h_stage2, 1 << 20));
  CK(hipMalloc(&d_raw, 64 << 20));
  CK(hipMalloc((void**)&d_raws, 1 << 19));
  CK(hipMalloc((void**)&d_cb, 64));

  CK(hipMalloc((void**)&d_embed, (long)VOC * HID * 2));
  fprintf(stderr, "trunk: loading embed...\n");
  load_bf16("embed_tokens.weight", (long)VOC * HID * 2, d_embed);
  CK(hipMalloc((void**)&d_lmhead, (long)VOC * HID * 2));
  fprintf(stderr, "trunk: loading lm_head (fp8r dequant)...\n");
  load_fp8r("lm_head.weight", VOC, HID, d_lmhead);
  CK(hipDeviceSynchronize());
  fprintf(stderr, "trunk: big tensors loaded (%.1fs)\n", now_s() - t0);

  CK(hipMalloc((void**)&d_ln_in_all, (long)NL * HID * 2));
  CK(hipMalloc((void**)&d_ln_post_all, (long)NL * HID * 2));
  for (int l = 0; l < NL; l++) {
    snprintf(nm, sizeof(nm), "layers.%d.input_layernorm.weight", l);
    load_bf16(nm, HID * 2, d_ln_in_all + (long)l * HID);
    snprintf(nm, sizeof(nm), "layers.%d.post_attention_layernorm.weight", l);
    load_bf16(nm, HID * 2, d_ln_post_all + (long)l * HID);
  }
  CK(hipMalloc((void**)&d_final_norm, HID * 2)); load_bf16("norm.weight", HID * 2, d_final_norm);
  CK(hipMalloc((void**)&d_gdn_conv, (long)48 * QKVW * 4 * 2));
  CK(hipMalloc((void**)&d_gdn_a, (long)48 * HV * HID * 2));
  CK(hipMalloc((void**)&d_gdn_b, (long)48 * HV * HID * 2));
  CK(hipMalloc((void**)&d_gdn_norm, (long)48 * DV * 2));
  h_alog = (float*)malloc((long)48 * HV * 4);
  h_dtbias = (float*)malloc((long)48 * HV * 4);
  { int gi = 0;
    for (int l = 0; l < NL; l++) {
      if (l % 4 == 3) continue;
      snprintf(nm, sizeof(nm), "layers.%d.linear_attn.conv1d.weight", l);
      load_bf16(nm, (long)QKVW * 4 * 2, d_gdn_conv + (long)gi * QKVW * 4);
      snprintf(nm, sizeof(nm), "layers.%d.linear_attn.in_proj_a.weight", l);
      load_bf16(nm, (long)HV * HID * 2, d_gdn_a + (long)gi * HV * HID);
      snprintf(nm, sizeof(nm), "layers.%d.linear_attn.in_proj_b.weight", l);
      load_bf16(nm, (long)HV * HID * 2, d_gdn_b + (long)gi * HV * HID);
      snprintf(nm, sizeof(nm), "layers.%d.linear_attn.norm.weight", l);
      load_bf16(nm, DV * 2, d_gdn_norm + (long)gi * DV);
      const Ent* e = find((snprintf(nm, sizeof(nm), "layers.%d.linear_attn.A_log", l), nm));
      rd(e->off, h_stage2, 96);
      { uint16_t* u = (uint16_t*)h_stage2;
        for (int i = 0; i < HV; i++) h_alog[(long)gi * HV + i] = h_bf16f(u[i]); }
      e = find((snprintf(nm, sizeof(nm), "layers.%d.linear_attn.dt_bias", l), nm));
      rd(e->off, h_stage2, 96);
      { uint16_t* u = (uint16_t*)h_stage2;
        for (int i = 0; i < HV; i++) h_dtbias[(long)gi * HV + i] = h_bf16f(u[i]); }
      gi++;
    } }
  CK(hipMalloc((void**)&d_qnorm, (long)16 * HD * 2));
  CK(hipMalloc((void**)&d_knorm, (long)16 * HD * 2));
  { int fi = 0;
    for (int l = 0; l < NL; l++) {
      if (l % 4 != 3) continue;
      snprintf(nm, sizeof(nm), "layers.%d.self_attn.q_norm.weight", l); load_bf16(nm, HD * 2, d_qnorm + (long)fi * HD);
      snprintf(nm, sizeof(nm), "layers.%d.self_attn.k_norm.weight", l); load_bf16(nm, HD * 2, d_knorm + (long)fi * HD);
      fi++;
    } }
  fprintf(stderr, "trunk: small tensors loaded\n");

  CK(hipMalloc((void**)&d_qkv, (long)QKVW * HID * 2));
  CK(hipMalloc((void**)&d_z, (long)HV * DV * HID * 2));
  CK(hipMalloc((void**)&d_outp, (long)HID * HV * DV * 2));
  CK(hipMalloc((void**)&d_q, (long)QH * HD * 2 * HID * 2));
  CK(hipMalloc((void**)&d_k, (long)KVH * HD * HID * 2));
  CK(hipMalloc((void**)&d_v, (long)KVH * HD * HID * 2));
  CK(hipMalloc((void**)&d_o, (long)HID * QH * HD * 2));
  CK(hipMalloc((void**)&d_gate, (long)INTM * HID * 2));
  CK(hipMalloc((void**)&d_up, (long)INTM * HID * 2));
  CK(hipMalloc((void**)&d_down, (long)HID * INTM * 2));
  for (int i = 0; i < 3; i++) {
    CK(hipMalloc((void**)&d_wc[i], (long)INTM * HID / 2 + 1024));
    CK(hipMalloc((void**)&d_ws[i], (long)INTM * HID / 256 * 2 + 1024));
  }
  CK(hipMalloc((void**)&d_qc, (long)TMAX * INTM / 2 + 1024));
  CK(hipMalloc((void**)&d_qs, (long)TMAX * (INTM / 256) * 2 + 1024));

  CK(hipMalloc((void**)&d_convstate, (long)48 * QKVW * 3 * 2));
  CK(hipMemset(d_convstate, 0, (long)48 * QKVW * 3 * 2));
  CK(hipMalloc((void**)&d_kcache, (long)16 * CXT * KVH * HD * 2));
  CK(hipMalloc((void**)&d_vcache, (long)16 * CXT * KVH * HD * 2));
  CK(hipMemset(d_kcache, 0, (long)16 * CXT * KVH * HD * 2));
  CK(hipMemset(d_vcache, 0, (long)16 * CXT * KVH * HD * 2));
  h_S = (float*)calloc((size_t)48 * HV * DK * DV, 4);

  CK(hipMalloc((void**)&d_x, (long)TMAX * HID * 2));
  CK(hipMalloc((void**)&d_ln1, (long)TMAX * HID * 2));
  CK(hipMalloc((void**)&d_ln2, (long)TMAX * HID * 2));
  CK(hipMalloc((void**)&d_tmp1, (long)TMAX * HID * 2));
  CK(hipMalloc((void**)&d_qkvraw, (long)TMAX * QKVW * 2));
  CK(hipMalloc((void**)&d_qkvc, (long)TMAX * QKVW * 2));
  CK(hipMalloc((void**)&d_ab_a, (long)TMAX * HV * 2));
  CK(hipMalloc((void**)&d_ab_b, (long)TMAX * HV * 2));
  CK(hipMalloc((void**)&d_zout, (long)TMAX * HV * DV * 2));
  CK(hipMalloc((void**)&d_co, (long)TMAX * HV * DV * 2));
  CK(hipMalloc((void**)&d_gn, (long)TMAX * HV * DV * 2));
  CK(hipMalloc((void**)&d_qg, (long)TMAX * QH * HD * 2 * 2));
  CK(hipMalloc((void**)&d_kk, (long)TMAX * KVH * HD * 2));
  CK(hipMalloc((void**)&d_vv, (long)TMAX * KVH * HD * 2));
  CK(hipMalloc((void**)&d_att, (long)TMAX * QH * HD * 2));
  CK(hipMalloc((void**)&d_qn, (long)TMAX * QH * HD * 2));
  CK(hipMalloc((void**)&d_kn, (long)TMAX * KVH * HD * 2));
  CK(hipMalloc((void**)&d_g17, (long)TMAX * INTM * 2));
  CK(hipMalloc((void**)&d_u17, (long)TMAX * INTM * 2));
  CK(hipMalloc((void**)&d_act, (long)TMAX * INTM * 2));
  CK(hipMalloc((void**)&d_scores, (long)QH * CXT * CXT * 4));
  CK(hipMalloc((void**)&d_pbf, (long)QH * CXT * CXT * 2));
  CK(hipMalloc((void**)&d_logits, (long)TMAX * VOC * 4));
  CK(hipMalloc((void**)&d_tok, TMAX * 4));
  CK(hipMalloc((void**)&d_tgt, TMAX * 4));
  CK(hipMalloc((void**)&d_pred, TMAX * 4));
  CK(hipMalloc((void**)&d_nll, TMAX * 4));

  h_qkvbf = (uint16_t*)malloc((long)TMAX * QKVW * 2);
  h_abbf = (uint16_t*)malloc((long)TMAX * HV * 2);
  h_af = (float*)malloc((long)TMAX * HV * 4);
  h_bf = (float*)malloc((long)TMAX * HV * 4);
  h_co = (uint16_t*)malloc((long)TMAX * HV * DV * 2);
  h_qscale = 1.0f / sqrtf((float)DK);

  // device weight pool: all quantized projection payloads resident; forward()
  // dequants from device memory per use (fallback: per-use file streaming)
  {
    size_t need = 0;
    for (int l = 0; l < NL; l++) {
      if (l % 4 == 3) {
        need += find((snprintf(nm, sizeof(nm), "layers.%d.self_attn.q_proj.weight", l), nm))->size;
        need += find((snprintf(nm, sizeof(nm), "layers.%d.self_attn.k_proj.weight", l), nm))->size;
        need += find((snprintf(nm, sizeof(nm), "layers.%d.self_attn.v_proj.weight", l), nm))->size;
        need += find((snprintf(nm, sizeof(nm), "layers.%d.self_attn.o_proj.weight", l), nm))->size;
      } else {
        need += find((snprintf(nm, sizeof(nm), "layers.%d.linear_attn.in_proj_qkv.weight", l), nm))->size;
        need += find((snprintf(nm, sizeof(nm), "layers.%d.linear_attn.in_proj_z.weight", l), nm))->size;
        need += find((snprintf(nm, sizeof(nm), "layers.%d.linear_attn.out_proj.weight", l), nm))->size;
      }
      need += find((snprintf(nm, sizeof(nm), "layers.%d.mlp.gate_proj.weight", l), nm))->size;
      need += find((snprintf(nm, sizeof(nm), "layers.%d.mlp.up_proj.weight", l), nm))->size;
      need += find((snprintf(nm, sizeof(nm), "layers.%d.mlp.down_proj.weight", l), nm))->size;
    }
    hipError_t pe = hipMalloc(&d_wpool, need);
    if (pe != hipSuccess) {
      // APUs expose system RAM to the GPU; pinned host memory is the bigger pool
      pe = hipHostMalloc(&d_wpool, need);
      if (pe != hipSuccess) { d_wpool = nullptr; g_wpool_cap = 0; }
    }
    if (d_wpool) {
      g_wpool_cap = need;
      bool ok = true;
      for (int l = 0; l < NL && ok; l++) {
        if (l % 4 == 3) {
          ok &= pool_add((snprintf(nm, sizeof(nm), "layers.%d.self_attn.q_proj.weight", l), nm), QH * HD * 2, HID);
          ok &= pool_add((snprintf(nm, sizeof(nm), "layers.%d.self_attn.k_proj.weight", l), nm), KVH * HD, HID);
          ok &= pool_add((snprintf(nm, sizeof(nm), "layers.%d.self_attn.v_proj.weight", l), nm), KVH * HD, HID);
          ok &= pool_add((snprintf(nm, sizeof(nm), "layers.%d.self_attn.o_proj.weight", l), nm), HID, QH * HD);
        } else {
          ok &= pool_add((snprintf(nm, sizeof(nm), "layers.%d.linear_attn.in_proj_qkv.weight", l), nm), QKVW, HID);
          ok &= pool_add((snprintf(nm, sizeof(nm), "layers.%d.linear_attn.in_proj_z.weight", l), nm), HV * DV, HID);
          ok &= pool_add((snprintf(nm, sizeof(nm), "layers.%d.linear_attn.out_proj.weight", l), nm), HID, HV * DV);
        }
        ok &= pool_add((snprintf(nm, sizeof(nm), "layers.%d.mlp.gate_proj.weight", l), nm), INTM, HID);
        ok &= pool_add((snprintf(nm, sizeof(nm), "layers.%d.mlp.up_proj.weight", l), nm), INTM, HID);
        ok &= pool_add((snprintf(nm, sizeof(nm), "layers.%d.mlp.down_proj.weight", l), nm), HID, INTM);
      }
      if (ok) fprintf(stderr, "trunk: weight pool resident (%.1f GB)\n", g_wpool_used / 1e9);
      else { hipFree(d_wpool); d_wpool = nullptr; g_wmap.clear(); g_wpool_used = 0;
             fprintf(stderr, "trunk: weight pool incomplete, streaming mode\n"); }
    } else {
      d_wpool = nullptr;
      fprintf(stderr, "trunk: weight pool alloc failed (%.1f GB), streaming mode\n", need / 1e9);
    }
  }
  g_ready = 1;
  fprintf(stderr, "trunk: ready (%.1fs)\n", now_s() - t0);
  return 0;
}

// Greedy generate: clean prefill, then per-step argmax with per-layer KV cache +
// GDN state advance. Emits each token via `emit`. Returns tokens emitted, or
// -1 (not ready) / -2 (prompt or max_tokens out of trunk capacity).
extern "C" int chlorine_trunk_generate(const int* prompt, int n_prompt, long max_tokens,
                                       const int* eos, int n_eos,
                                       void (*emit)(void*, int), void* ctx,
                                       double* prefill_ms, double* decode_ms, int* stop_hit) {
  if (!g_ready) return -1;
  if (n_prompt < 1 || n_prompt > CXT - 2 || max_tokens < 1 ||
      max_tokens > (long)(CXT - n_prompt)) return -2;
  trunk_reset_states();
  double tp = now_s();
  up(d_tok, prompt, (size_t)n_prompt * 4);
  forward(d_tok, n_prompt, 0, 1);
  *prefill_ms = (now_s() - tp) * 1000.0;

  double td = now_s();
  int n_gen = 0, tok = 0, stop = 0;
  k_argmax1<<<1, 256>>>(d_logits + (long)(n_prompt - 1) * VOC, d_pred, VOC);
  CK(hipDeviceSynchronize());
  CK(hipMemcpy(&tok, d_pred, 4, hipMemcpyDeviceToHost));
  while (!stop && n_gen < max_tokens) {
    emit(ctx, tok);
    n_gen++;
    for (int i = 0; i < n_eos; i++)
      if (eos[i] == tok) { stop = 1; break; }
    if (stop || n_gen >= max_tokens) break;
    up(d_tok, &tok, 4);
    forward(d_tok, 1, n_prompt - 1 + n_gen, 1);
    k_argmax1<<<1, 256>>>(d_logits, d_pred, VOC);
    CK(hipDeviceSynchronize());
    CK(hipMemcpy(&tok, d_pred, 4, hipMemcpyDeviceToHost));
  }
  *decode_ms = (now_s() - td) * 1000.0;
  *stop_hit = stop;
  return n_gen;
}

// ---------------- sampler (HOST-LOGIC.md section 5) ----------------
// k_sample semantics: penalties -> bf16 logits -> temp softmax -> top-k ->
// top-p -> min-p -> renormalize -> draw. Counter RNG decoded from the k_sample
// ISA (obj1.so): c = posctr*0xd1b54a32d192ed03 ^ seed ^ (aux*0x9e3779b97f4a7c15);
// c += 0x9e3779b97f4a7c15; splitmix64 finalizer; u = (c >> 40) * 2^-24.
typedef struct {
  int has_sample;
  float temp;
  int top_k;
  float top_p, min_p;
  unsigned long long seed;
  float presence, frequency;
  int n_bias;
  const int* bias_ids;
  const float* bias_vals;
  int logprobs;
} chlorine_sample_opts;

static float* g_pinrow = nullptr;   // pinned VOC f32 staging for sampler copies

static inline unsigned long long sm_fin(unsigned long long x) {
  x ^= x >> 30; x *= 0xbf58476d1ce4e5b9ULL;
  x ^= x >> 27; x *= 0x94d049bb133111ebULL;
  return x;
}

static float sample_uniform(unsigned long long seed, unsigned int posctr, unsigned int aux) {
  unsigned long long c = (unsigned long long)posctr * 0xd1b54a32d192ed03ULL;
  c ^= (unsigned long long)aux * 0x9e3779b97f4a7c15ULL;
  c ^= seed;
  c += 0x9e3779b97f4a7c15ULL;
  c = sm_fin(c);
  return (float)(unsigned)(c >> 40) * (1.0f / 16777216.0f);
}

// One sampling step. d_row: device f32 logits [VOC]; counts: per-request
// generated-token counts (host, VOC); posctr: n_prompt + generated-so-far.
extern "C" int chlorine_sample_host(const float* d_row, int* counts, unsigned int posctr,
                                    const chlorine_sample_opts* o,
                                    int* out_tok, float* out_logp) {
  if (!g_pinrow) CK(hipHostMalloc((void**)&g_pinrow, (size_t)VOC * 4));
  CK(hipMemcpy(g_pinrow, d_row, (size_t)VOC * 4, hipMemcpyDeviceToHost));
  float* L = g_pinrow;
  // engine logits are bf16 (k_nll/k_argmax read u16); penalties scatter into a
  // bf16 copy, so round first and re-round after every f32 delta add
  for (int v = 0; v < VOC; v++) L[v] = h_bf16f(h_f32bf(L[v]));
  if (o->presence != 0.f || o->frequency != 0.f) {
    for (int t = 0; t < VOC; t++) {
      if (counts[t] > 0) {
        float d = -(o->presence + o->frequency * (float)counts[t]);
        L[t] = h_bf16f(h_f32bf(L[t] + d));
      }
    }
  }
  for (int i = 0; i < o->n_bias; i++) {
    int t = o->bias_ids[i];
    if (t >= 0 && t < VOC) L[t] = h_bf16f(h_f32bf(L[t] + o->bias_vals[i]));
  }
  // temp softmax (f32 throughout, like k_sample)
  float mx = L[0];
  for (int v = 1; v < VOC; v++) if (L[v] > mx) mx = L[v];
  float inv_t = 1.0f / o->temp;
  static std::vector<float> p;
  static std::vector<unsigned char> keep;
  static std::vector<int> idx;
  p.resize(VOC); keep.resize(VOC); idx.resize(VOC);
  double tot = 0.0;
  for (int v = 0; v < VOC; v++) { float e = expf((L[v] - mx) * inv_t); p[v] = e; tot += e; }
  float ftot = (float)tot;
  int nkeep = 0;
  for (int v = 0; v < VOC; v++) idx[v] = v;
  if (o->top_k > 0 && o->top_k < VOC) {
    // top-k: k largest probs (ties -> lower id), via nth_element
    int k = o->top_k;
    std::nth_element(idx.data(), idx.data() + k, idx.data() + VOC,
                     [&](int a, int b) {
                       if (p[a] != p[b]) return p[a] > p[b];
                       return a < b;
                     });
    for (int i = 0; i < k; i++) keep[idx[i]] = 1;
    nkeep = k;
  } else {
    for (int v = 0; v < VOC; v++) keep[v] = 1;
    nkeep = VOC;
  }
  if (o->top_p > 0.f && o->top_p < 1.f) {
    // nucleus within the kept set: sort desc, keep until cum >= top_p (inclusive)
    std::vector<int> nk;
    nk.reserve(nkeep);
    for (int v = 0; v < VOC; v++) if (keep[v]) nk.push_back(v);
    std::sort(nk.begin(), nk.end(), [&](int a, int b) {
      if (p[a] != p[b]) return p[a] > p[b];
      return a < b;
    });
    float cum = 0.f;
    int cut = (int)nk.size();
    for (int i = 0; i < (int)nk.size(); i++) {
      cum += p[nk[i]];
      if (cum >= o->top_p * ftot) { cut = i + 1; break; }
    }
    for (int i = 0; i < cut; i++) keep[nk[i]] = 1;
    for (int i = cut; i < (int)nk.size(); i++) keep[nk[i]] = 0;
  }
  if (o->min_p > 0.f) {
    // min-p: normalized prob >= min_p * p_max (p_max = 1/tot after temp scaling)
    float thr = o->min_p / ftot;
    for (int v = 0; v < VOC; v++)
      if (keep[v] && p[v] < thr) keep[v] = 0;
  }
  double tkept = 0.0;
  for (int v = 0; v < VOC; v++) if (keep[v]) tkept += p[v];
  if (tkept <= 0.0) {  // degenerate: fall back to argmax
    int am = 0;
    for (int v = 1; v < VOC; v++) if (L[v] > L[am]) am = v;
    *out_tok = am;
    *out_logp = 0.f;
    return 0;
  }
  float u = sample_uniform(o->seed, posctr, 0);
  float target = u * (float)tkept;
  // the engine scans the support in PROB-DESC order (verified: seeded coin A/B
  // vs the original server — 92/100 bit agreement with the sorted-order model)
  static std::vector<int> sorder;
  sorder.clear();
  for (int v = 0; v < VOC; v++)
    if (keep[v]) sorder.push_back(v);
  std::sort(sorder.begin(), sorder.end(), [&](int a, int b) {
    if (p[a] != p[b]) return p[a] > p[b];
    return a < b;
  });
  float cum = 0.f;
  int tok = -1;
  for (int v : sorder) {
    cum += p[v];
    if (cum >= target) { tok = v; break; }
  }
  if (tok < 0 && !sorder.empty()) tok = sorder.back();
  *out_tok = tok;
  *out_logp = logf(p[tok] / (float)tkept);
  return 0;
}

// Sampling generate: same trunk loop as chlorine_trunk_generate with the host
// sampler (temp/top-k/top-p/min-p/seed, presence/frequency penalties, bias,
// logprobs). posctr for the counter RNG = n_prompt + generated-so-far (the
// engine passes base + loop index; first token = n_prompt).
extern "C" int chlorine_trunk_generate2(const int* prompt, int n_prompt, long max_tokens,
                                        const int* eos, int n_eos,
                                        void (*emit)(void*, int, float), void* ctx,
                                        double* prefill_ms, double* decode_ms, int* stop_hit,
                                        const chlorine_sample_opts* opts) {
  if (!g_ready) return -1;
  if (n_prompt < 1 || n_prompt > CXT - 2 || max_tokens < 1 ||
      max_tokens > (long)(CXT - n_prompt)) return -2;
  bool sample = opts && opts->has_sample;
  static std::vector<int> counts;
  counts.assign(VOC, 0);
  trunk_reset_states();
  double tp = now_s();
  up(d_tok, prompt, (size_t)n_prompt * 4);
  forward(d_tok, n_prompt, 0, 1);
  *prefill_ms = (now_s() - tp) * 1000.0;

  double td = now_s();
  int n_gen = 0, tok = 0, stop = 0;
  float logp = 0.f;
  if (sample) {
    chlorine_sample_host(d_logits + (long)(n_prompt - 1) * VOC, counts.data(),
                         (unsigned)n_prompt, opts, &tok, &logp);
    counts[tok]++;
  } else {
    k_argmax1<<<1, 256>>>(d_logits + (long)(n_prompt - 1) * VOC, d_pred, VOC);
    CK(hipDeviceSynchronize());
    CK(hipMemcpy(&tok, d_pred, 4, hipMemcpyDeviceToHost));
  }
  while (!stop && n_gen < max_tokens) {
    emit(ctx, tok, sample ? logp : 0.f);
    n_gen++;
    for (int i = 0; i < n_eos; i++)
      if (eos[i] == tok) { stop = 1; break; }
    if (stop || n_gen >= max_tokens) break;
    up(d_tok, &tok, 4);
    forward(d_tok, 1, n_prompt - 1 + n_gen, 1);
    if (sample) {
      chlorine_sample_host(d_logits, counts.data(), (unsigned)(n_prompt + n_gen),
                           opts, &tok, &logp);
      counts[tok]++;
    } else {
      k_argmax1<<<1, 256>>>(d_logits, d_pred, VOC);
      CK(hipDeviceSynchronize());
      CK(hipMemcpy(&tok, d_pred, 4, hipMemcpyDeviceToHost));
    }
  }
  *decode_ms = (now_s() - td) * 1000.0;
  *stop_hit = stop;
  return n_gen;
}

extern "C" void chlorine_trunk_shutdown(void) {
  // weights and buffers live for the process lifetime; nothing to do
}
