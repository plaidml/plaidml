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

bool IsIntegerDataType(DataType type) {
  return type == DataType::INT128 ||
         type == DataType::INT64 ||
         type == DataType::INT32 ||
         type == DataType::INT16 ||
         type == DataType::INT8;
}

bool IsFloatDataType(DataType type) {
  return type == DataType::FLOAT64 ||
         type == DataType::FLOAT32 ||
         type == DataType::FLOAT16 ||
         type == DataType::BFLOAT16;
}

int64_t IntegerMax(DataType type) {
  if (type == DataType::INT64) {
    return (int64_t)std::numeric_limits<int64_t>::max();
  }
  if (type == DataType::INT32) {
    return (int64_t)std::numeric_limits<int32_t>::max();
  }
  if (type == DataType::INT16) {
    return (int64_t)std::numeric_limits<int16_t>::max();
  }
  if (type == DataType::INT8) {
    return (int64_t)std::numeric_limits<int8_t>::max();
  }
  throw std::runtime_error("Unsupported integer type for max value");
}

int64_t IntegerMin(DataType type) {
  if (type == DataType::INT64) {
    return (int64_t)std::numeric_limits<int64_t>::lowest();
  }
  if (type == DataType::INT32) {
    return (int64_t)std::numeric_limits<int32_t>::lowest();
  }
  if (type == DataType::INT16) {
    return (int64_t)std::numeric_limits<int16_t>::lowest();
  }
  if (type == DataType::INT8) {
    return (int64_t)std::numeric_limits<int8_t>::lowest();
  }
  throw std::runtime_error("Unsupported integer type for min value");
}

double FloatMax(DataType type) {
  if (type == DataType::FLOAT64) {
    return (double)std::numeric_limits<double>::max();
  }
  if (type == DataType::FLOAT32) {
    return (double)std::numeric_limits<float>::max();
  }
  throw std::runtime_error("Unsupported float type for max value");
}

double FloatMin(DataType type) {
  if (type == DataType::FLOAT64) {
    return (double)std::numeric_limits<double>::lowest();
  }
  if (type == DataType::FLOAT32) {
    return (double)std::numeric_limits<float>::lowest();
  }
  throw std::runtime_error("Unsupported float type for min value");
}

}  // namespace tile
}  // namespace vertexai
