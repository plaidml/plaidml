// Copyright 2018 Intel Corporation.

#include "tile/base/shape.h"

namespace vertexai {
namespace tile {

std::ostream& operator<<(std::ostream& os, const TensorShape& shape) {
  os << to_string(shape.type) << "(";
  for (size_t i = 0; i < shape.dims.size(); i++) {
    if (i > 0) {
      os << ", ";
    }
    os << shape.dims[i].size;
  }
  os << "):(";
  for (size_t i = 0; i < shape.dims.size(); i++) {
    if (i > 0) {
      os << ", ";
    }
    os << shape.dims[i].stride;
  }
  os << "):";
  if (shape.byte_size() < 1024) {
    os << shape.byte_size() << " bytes";
  } else {
    os << shape.byte_size() / 1024.0 << " KiB";
  }
  return os;
}

std::ostream& operator<<(std::ostream& os, const TensorDimension& dim) {
  os << dim.size << ":" << dim.stride;
  return os;
}

void TensorShape::resize_dim(size_t pos, uint64_t size) {
  assert(pos < dims.size());
  dims[pos].size = size;
  std::multimap<int64_t, TensorDimension*> sorted;
  for (auto& dim : dims) {
    sorted.emplace(dim.stride, &dim);
  }
  int64_t stride = 1;
  for (auto& item : sorted) {
    item.second->stride = stride;
    stride *= item.second->size;
  }
}

}  // namespace tile
}  // namespace vertexai
