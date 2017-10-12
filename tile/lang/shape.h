#pragma once

#include <map>
#include <string>
#include <utility>
#include <vector>

#include "base/util/logging.h"

namespace vertexai {
namespace tile {
namespace lang {

enum class DataType : int {
  INVALID = 0,
  BOOLEAN = 0x02,
  INT8 = 0x10,
  INT16 = 0x11,
  INT32 = 0x12,
  INT64 = 0x13,
  UINT8 = 0x20,
  UINT16 = 0x21,
  UINT32 = 0x22,
  UINT64 = 0x23,
  FLOAT16 = 0x31,
  FLOAT32 = 0x32,
  FLOAT64 = 0x33,
  PRNG = 0x40,
};

inline bool is_float(const DataType& dt) {
  switch (dt) {
    case DataType::FLOAT16:
    case DataType::FLOAT32:
    case DataType::FLOAT64:
      return true;
    default:
      return false;
  }
}

inline size_t bit_width(const DataType& dt) {
  switch (dt) {
    case DataType::BOOLEAN:
      return 8;
    case DataType::INT8:
      return 8;
    case DataType::INT16:
      return 16;
    case DataType::INT32:
      return 32;
    case DataType::INT64:
      return 64;
    case DataType::UINT8:
      return 8;
    case DataType::UINT16:
      return 16;
    case DataType::UINT32:
      return 32;
    case DataType::UINT64:
      return 64;
    case DataType::FLOAT16:
      return 16;
    case DataType::FLOAT32:
      return 32;
    case DataType::FLOAT64:
      return 64;
    default:
      return 0;
  }
}

inline std::string to_string(const DataType& dt) {
  switch (dt) {
    case DataType::BOOLEAN:
      return "boolean";
    case DataType::INT8:
      return "int8";
    case DataType::INT16:
      return "int16";
    case DataType::INT32:
      return "int32";
    case DataType::INT64:
      return "int64";
    case DataType::UINT8:
      return "uint8";
    case DataType::UINT16:
      return "uint16";
    case DataType::UINT32:
      return "uint32";
    case DataType::UINT64:
      return "uint64";
    case DataType::FLOAT16:
      return "float16";
    case DataType::FLOAT32:
      return "float32";
    case DataType::FLOAT64:
      return "float64";
    case DataType::PRNG:
      return "prng";
    default:
      return "!!invalid data type";
  }
}

inline size_t byte_width(const DataType& dt) { return (bit_width(dt) + 7) / 8; }

struct TensorDimension {
  TensorDimension() = default;
  TensorDimension(int64_t _stride, uint64_t _size) : stride(_stride), size(_size) {}
  // Stride over element count
  int64_t stride;
  // Number of elements
  uint64_t size;
  // Comparison operators
  bool operator==(const TensorDimension& rhs) const { return (stride == rhs.stride && size == rhs.size); }
  bool operator<(const TensorDimension& rhs) const {
    return std::make_pair(stride, size) < std::make_pair(rhs.stride, rhs.size);
  }
};

struct TensorShape {
  TensorShape() = default;
  TensorShape(DataType _type, const std::vector<TensorDimension>& _dims) : type(_type), dims(_dims) {}
  DataType type = DataType::INVALID;
  std::vector<TensorDimension> dims;
  bool operator==(const TensorShape& rhs) const { return type == rhs.type && dims == rhs.dims; }
  bool operator<(const TensorShape& rhs) const {
    return std::make_pair(type, dims) < std::make_pair(rhs.type, rhs.dims);
  }
  uint64_t buffer_size() const {
    uint64_t max_elem = 0;
    for (const auto& dim : dims) {
      if (dim.stride > 0) {
        max_elem += (dim.size - 1) * dim.stride;
      }
    }
    return max_elem + 1;
  }
};

inline MAKE_LOGGABLE(TensorShape, ts, os) {
  os << to_string(ts.type) << "(";
  for (size_t i = 0; i < ts.dims.size(); i++) {
    os << ts.dims[i].size << ":" << ts.dims[i].stride;
    if (i != ts.dims.size() - 1) {
      os << ", ";
    }
  }
  os << ")";
  return os;
}

inline TensorShape SimpleShape(DataType type, const std::vector<size_t>& sizes) {
  int64_t stride = 1;
  std::vector<TensorDimension> dims(sizes.size());
  for (int i = sizes.size() - 1; i >= 0; i--) {
    dims[i].stride = stride;
    dims[i].size = sizes[i];
    stride *= sizes[i];
  }
  return TensorShape(type, dims);
}

typedef std::map<std::string, TensorShape> ShapeMap;

}  // namespace lang
}  // namespace tile
}  // namespace vertexai
