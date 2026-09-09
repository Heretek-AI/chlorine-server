// hgn.h — .hgn checkpoint loader (spec: docs/halogen/CHECKPOINT-FORMAT.md)
#pragma once
#include <cstdint>
#include <string>
#include <string_view>
#include <unordered_map>
#include <vector>

namespace chlorine {

struct Tensor {
  std::string name;
  uint64_t f1 = 0;      // (ndim << 32) | dtype_id
  uint64_t dims[4] = {};
  uint64_t offset = 0;  // data offset in file
  uint64_t size = 0;    // data bytes
  uint64_t qparam = 0;  // f8 >> 32
  uint32_t dtype() const { return uint32_t(f1 & 0xFFFFFFFFu); }
  uint32_t ndim() const { return uint32_t(f1 >> 32); }
};

enum DType : uint32_t {
  kBF16 = 0, kF32 = 1, kF16 = 2, kI32 = 3, kI64 = 4,
  kQ4C = 5, kFP8R = 6, kQ8G64 = 7, kI4L = 8,
};

inline const char* dtype_name(uint32_t d) {
  static const char* kNames[] = {"bf16", "f32", "f16", "i32", "i64",
                                 "q4c", "fp8r", "q8g64", "i4l"};
  return d < 9 ? kNames[d] : "unknown";
}

class Checkpoint {
 public:
  // Opens, validates and mmaps the file. Dies with the engine's error
  // messages on spec violations (bad magic / version / size / bounds).
  explicit Checkpoint(const std::string& path);
  ~Checkpoint();

  const Tensor* find(std::string_view name) const;
  const uint8_t* data(const Tensor& t) const {
    return base_ + t.offset;
  }
  const std::string& model_name() const { return model_name_; }
  uint64_t n_tensors() const { return tensors_.size(); }
  uint64_t file_size() const { return file_size_; }

  // Capability flags derived from tensor presence (drives the INFO line).
  bool has_embed() const { return find("embed_tokens.weight") != nullptr; }
  bool has_lm_head() const { return find("lm_head.weight") != nullptr; }
  bool has_mtp() const { return find("mtp.fc.weight") != nullptr; }
  bool has_dflash2() const { return find("drafter.fc.weight") != nullptr; }

 private:
  uint8_t* base_ = nullptr;
  uint64_t file_size_ = 0;
  std::string model_name_;
  std::vector<Tensor> tensors_;
  std::unordered_map<std::string, size_t> index_;
};

}  // namespace chlorine
