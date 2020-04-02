// Copyright 2020 Intel Corporation

#include <stddef.h>

#include "llvm/Support/raw_ostream.h"

#include "pmlc/rt/memref.h"

extern "C" void plaidml_rt_trace(const char *msg) {
  llvm::outs() << msg << "\n";
  llvm::outs().flush();
}

namespace {
template <typename T>
void plaidml_rt_copy(StridedMemRefType<T, 0> *dest,
                     StridedMemRefType<T, 0> *src, int32_t count) {
  auto destPtr = dest->data + dest->offset;
  auto srcPtr = src->data + src->offset;
  size_t bytes = count * sizeof(T);
  memcpy(destPtr, srcPtr, bytes);
}
} // namespace

extern "C" void plaidml_rt_copy_f32(size_t destRank,
                                    StridedMemRefType<float, 0> *dest,
                                    size_t srcRank,
                                    StridedMemRefType<float, 0> *src,
                                    int32_t count) {
  plaidml_rt_copy<float>(dest, src, count);
}
