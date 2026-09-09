// hgn.cpp — .hgn checkpoint loader.
// Behavior mirrors the original engine's loader (checkpoint.cpp) including
// its error strings, but is an independent implementation from the spec.
#include "hgn.h"

#include <fcntl.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <unistd.h>

#include <cstdio>
#include <cstdlib>
#include <cstring>

namespace chlorine {
namespace {

[[noreturn]] void die(const std::string& msg) {
  fprintf(stderr, "%s\n", msg.c_str());
  exit(1);
}

}  // namespace

Checkpoint::Checkpoint(const std::string& path) {
  int fd = open(path.c_str(), O_RDONLY);
  if (fd < 0) die("checkpoint: cannot open " + path);
  struct stat st {};
  if (fstat(fd, &st) != 0) die("checkpoint: fstat failed on " + path);
  file_size_ = uint64_t(st.st_size);
  void* m = mmap(nullptr, file_size_, PROT_READ, MAP_PRIVATE, fd, 0);
  close(fd);
  if (m == MAP_FAILED) die("checkpoint: mmap failed on " + path);
  base_ = static_cast<uint8_t*>(m);

  if (file_size_ < 0x68) die("checkpoint: bad magic in " + path);
  uint32_t magic;
  memcpy(&magic, base_, 4);
  if (magic != 0x314E4748u) die("checkpoint: bad magic in " + path);
  uint32_t version;
  memcpy(&version, base_ + 4, 4);
  if (version - 1 > 1) die("checkpoint: unsupported version in " + path);
  uint64_t n, table_off, reserved, size_field;
  memcpy(&n, base_ + 8, 8);
  memcpy(&table_off, base_ + 0x10, 8);
  memcpy(&reserved, base_ + 0x18, 8);
  memcpy(&size_field, base_ + 0x20, 8);
  (void)reserved;  // first data offset; not needed here
  if (size_field != file_size_) die("checkpoint: truncated file " + path + " (size mismatch)");
  model_name_.assign(reinterpret_cast<const char*>(base_ + 0x28),
                     strnlen(reinterpret_cast<const char*>(base_ + 0x28), 64));

  tensors_.resize(n);
  const uint8_t* tbl = base_ + table_off;
  for (uint64_t i = 0; i < n; i++) {
    const uint8_t* e = tbl + i * 0xA0;
    Tensor& t = tensors_[i];
    t.name.assign(reinterpret_cast<const char*>(e),
                  strnlen(reinterpret_cast<const char*>(e), 96));
    memcpy(&t.f1, e + 0x60, 8);
    memcpy(&t.dims, e + 0x68, 32);
    memcpy(&t.offset, e + 0x88, 8);
    memcpy(&t.size, e + 0x90, 8);
    uint64_t f8;
    memcpy(&f8, e + 0x98, 8);
    t.qparam = uint32_t(f8 >> 32);
    if (t.offset + t.size > file_size_) die("checkpoint: entry out of bounds in " + path);
    index_[t.name] = i;
  }
}

Checkpoint::~Checkpoint() { munmap(base_, file_size_); }

const Tensor* Checkpoint::find(std::string_view name) const {
  auto it = index_.find(std::string(name));
  return it == index_.end() ? nullptr : &tensors_[it->second];
}

}  // namespace chlorine
