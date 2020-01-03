// Copyright 2018 Intel Corporation.

#include "tile/base/shape.h"

#include "base/util/logging.h"

namespace vertexai {
namespace tile {

std::ostream& operator<<(std::ostream& os, const TensorShape& shape) {
  os << to_string(shape.type);
  if (!shape.layout.empty()) {
    os << "[" << shape.layout << "]";
  }
  os << "(";
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
  if (shape.is_const) {
    os << " const";
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

DataType CommonSupertype(DataType lhs, DataType rhs) {
  IVLOG(6, "CommonSupertype> " << to_string(lhs) << " : " << to_string(rhs));
  if (lhs == DataType::INVALID) {
    return rhs;
  }
  if (rhs == DataType::INVALID) {
    return lhs;
  }
  if (is_float(lhs) != is_float(rhs)) {
    if (is_float(rhs)) {
      return rhs;
    } else {
      return lhs;
    }
  }
  // TODO: This is a bit primitive; for example, it will pick
  // the first of "int32" or "float32".  We may want to make it
  // a bit more sophisticated.
  if (bit_width(rhs) > bit_width(lhs)) {
    return rhs;
  }
  return lhs;
}

}  // namespace tile
}  // namespace vertexai
