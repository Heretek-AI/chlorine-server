// dequant_kernels.hip — chlorine-server dequant (spec: docs/halogen/KERNELS-DEQUANT.md)
// fp8r: w = fp8e4m3(payload[r*cols+c]) * bf16_scale[row at tensor tail]
// q4c:  w = cb[nib] * fp8e4m3(scale[r][g]) per 16-elem group, row-interleaved
#include <hip/hip_runtime.h>
#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <vector>

#define CK(x) do { hipError_t e = (x); if (e) { printf("HIP err %s @%d\n", hipGetErrorString(e), __LINE__); exit(1);} } while (0)

__device__ inline float bf16_to_f32(uint16_t h) {
  uint32_t u = (uint32_t)h << 16;
  return __uint_as_float(u);
}

__device__ inline float dequant_fp8_e4m3(uint32_t b) {
  float s = (b & 0x80u) ? -1.0f : 1.0f;
  uint32_t e = (b >> 3) & 0xFu, m = b & 0x7u;
  if (e == 0) return s * (float)m * 0.125f * ldexpf(1.0f, -6);
  return s * (1.0f + (float)m * 0.125f) * ldexpf(1.0f, (int)e - 7);
}

// one thread = one output element
__global__ void k_dequant_fp8r(const uint8_t* __restrict__ payload,
                               const uint16_t* __restrict__ scales,
                               uint16_t* __restrict__ out,
                               long cols) {
  long i = blockIdx.x * (long)blockDim.x + threadIdx.x;
  long r = i / cols;
  float w = dequant_fp8_e4m3(payload[i]) * bf16_to_f32(scales[r]);
  // round to bf16 (RNE via f32 bits)
  uint32_t u = __float_as_uint(w);
  uint32_t lo = u & 0xFFFFu;
  u = (u >> 16) + ((lo > 0x7FFFu) | ((lo == 0x7FFFu) & ((u >> 16) & 1u)));
  out[i] = (uint16_t)(u & 0xFFFFu);
}

// one thread = one output element; per-row: [cols/2 payload][cols/16 e4m3 scales]
__global__ void k_dequant_q4c(const float* __restrict__ cb,
                              const uint8_t* __restrict__ rowbase,
                              uint16_t* __restrict__ out,
                              long cols) {
  long i = blockIdx.x * (long)blockDim.x + threadIdx.x;
  long r = i / cols, c = i % cols;
  const uint8_t* pay = rowbase + (long)r * (cols / 2 + cols / 16);
  uint8_t byte = pay[c / 2];
  uint32_t nib = (c & 1) ? ((byte >> 4) & 0xFu) : (byte & 0xFu);
  float v = cb[nib];  // sign-magnitude: cb[8..15] = negatives of cb[0..7]
  float sc = dequant_fp8_e4m3(pay[cols / 2 + c / 16]);
  float w = v * sc;
  uint32_t u = __float_as_uint(w);
  uint32_t lo = u & 0xFFFFu;
  u = (u >> 16) + ((lo > 0x7FFFu) | ((lo == 0x7FFFu) & ((u >> 16) & 1u)));
  out[i] = (uint16_t)(u & 0xFFFFu);
}

static uint16_t to_bf16(float f) {
  uint32_t u;
  memcpy(&u, &f, 4);
  uint32_t lo = u & 0xFFFFu;
  u = (u >> 16) + ((lo > 0x7FFFu) | ((lo == 0x7FFFu) & ((u >> 16) & 1u)));
  return (uint16_t)(u & 0xFFFFu);
}

int main() {
  const char* ckpt = "/home/ronin/Projects/models/halogen-qwen3.8-27b/qwen3.8-27b-p1w4d-d2.hgn";
  FILE* f = fopen(ckpt, "rb");
  if (!f) { printf("no ckpt\n"); return 1; }
  // read tensor table
  fseek(f, 0x68, SEEK_SET);
  const int N = 1352;
  std::vector<uint8_t> tbl(N * 0xA0);
  if (fread(tbl.data(), 1, tbl.size(), f) != tbl.size()) { printf("table read fail\n"); return 1; }
  auto find = [&](const char* name) -> std::pair<long, long> {
    for (int i = 0; i < N; i++) {
      const uint8_t* e = tbl.data() + i * 0xA0;
      char nm[97]; memcpy(nm, e, 96); nm[96] = 0;
      if (strcmp(nm, name) == 0) {
        long off, size;
        memcpy(&off, e + 0x88, 8); memcpy(&size, e + 0x90, 8);
        return {off, size};
      }
    }
    printf("tensor %s not found\n", name);
    exit(1);
  };

  const int ROWS = 64, COLS = 5120;
  // ---- fp8r: layers.0.linear_attn.in_proj_qkv.weight ----
  {
    auto [off, size] = find("layers.0.linear_attn.in_proj_qkv.weight");
    std::vector<uint8_t> pay((size_t)ROWS * COLS);
    std::vector<uint8_t> scales((size_t)ROWS * 2);
    fseek(f, off, SEEK_SET);
    if (fread(pay.data(), 1, pay.size(), f) != pay.size()) { printf("pay read\n"); return 1; }
    fseek(f, off + size - (long)ROWS * 2, SEEK_SET);
    if (fread(scales.data(), 1, scales.size(), f) != scales.size()) { printf("sc read\n"); return 1; }
    uint8_t *dpay; uint16_t *dsc, *dout;
    CK(hipMalloc(&dpay, pay.size()));
    CK(hipMalloc(&dsc, scales.size()));
    CK(hipMalloc(&dout, (size_t)ROWS * COLS * 2));
    CK(hipMemcpy(dpay, pay.data(), pay.size(), hipMemcpyHostToDevice));
    CK(hipMemcpy(dsc, scales.data(), scales.size(), hipMemcpyHostToDevice));
    long n = (long)ROWS * COLS;
    k_dequant_fp8r<<<(n + 255) / 256, 256>>>(dpay, dsc, dout, COLS);
    CK(hipDeviceSynchronize());
    std::vector<uint16_t> out((size_t)n);
    CK(hipMemcpy(out.data(), dout, out.size() * 2, hipMemcpyDeviceToHost));
    FILE* o = fopen("/tmp/deq_fp8r.bin", "wb");
    fwrite(out.data(), 2, out.size(), o);
    fclose(o);
    printf("fp8r: dumped %ld bf16\n", n);
    CK(hipFree(dpay)); CK(hipFree(dsc)); CK(hipFree(dout));
  }
  // ---- q4c: layers.0.mlp.down_proj.weight ----
  {
    auto [off, size] = find("layers.0.mlp.down_proj.weight");
    const int R = 16, C = 17408;
    std::vector<uint8_t> cb(64), rows((size_t)R * (C / 2 + C / 16));
    fseek(f, off, SEEK_SET);
    if (fread(cb.data(), 1, 64, f) != 64) { printf("cb read\n"); return 1; }
    fseek(f, off + 64, SEEK_SET);
    if (fread(rows.data(), 1, rows.size(), f) != rows.size()) { printf("rows read\n"); return 1; }
    float* dcb; uint8_t* drows; uint16_t* dout;
    CK(hipMalloc(&dcb, 64));
    CK(hipMalloc(&drows, rows.size()));
    CK(hipMalloc(&dout, (size_t)R * C * 2));
    CK(hipMemcpy(dcb, cb.data(), 64, hipMemcpyHostToDevice));
    CK(hipMemcpy(drows, rows.data(), rows.size(), hipMemcpyHostToDevice));
    long n = (long)R * C;
    k_dequant_q4c<<<(n + 255) / 256, 256>>>(dcb, drows, dout, C);
    CK(hipDeviceSynchronize());
    std::vector<uint16_t> out((size_t)n);
    CK(hipMemcpy(out.data(), dout, out.size() * 2, hipMemcpyDeviceToHost));
    FILE* o = fopen("/tmp/deq_q4c.bin", "wb");
    fwrite(out.data(), 2, out.size(), o);
    fclose(o);
    printf("q4c: dumped %ld bf16\n", n);
    CK(hipFree(dcb)); CK(hipFree(drows)); CK(hipFree(dout));
  }
  fclose(f);
  return 0;
}
