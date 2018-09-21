#pragma once

#include <algorithm>
#include <map>
#include <ostream>
#include <string>
#include <utility>
#include <vector>

#include "base/util/transfer_object.h"
#include "tile/proto/shape.pb.h"

namespace vertexai {
namespace tile {

typedef std::vector<uint64_t> TileShape;

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

inline bool is_int(const DataType& dt) {
  switch (dt) {
    case DataType::INT8:
    case DataType::INT16:
    case DataType::INT32:
    case DataType::INT64:
      return true;
    default:
      return false;
  }
}

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
  uint64_t byte_size() const { return elem_size() * byte_width(type); }
  uint64_t elem_size() const {
    uint64_t max_elem = 0;
    for (const auto& dim : dims) {
      if (dim.stride > 0) {
        max_elem += (dim.size - 1) * dim.stride;
      }
    }
    return max_elem + 1;
  }
};

inline std::ostream& operator<<(std::ostream& os, const TensorShape& shape) {
  os << to_string(shape.type) << "(";
  for (size_t i = 0; i < shape.dims.size(); i++) {
    os << shape.dims[i].size << ":" << shape.dims[i].stride;
    if (i != shape.dims.size() - 1) {
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

inline DataType FromProto(const proto::TensorShape_DataType& dt) {
  switch (dt) {
    case proto::TensorShape_DataType_BOOLEAN:
      return DataType::BOOLEAN;
    case proto::TensorShape_DataType_INT8:
      return DataType::INT8;
    case proto::TensorShape_DataType_INT16:
      return DataType::INT16;
    case proto::TensorShape_DataType_INT32:
      return DataType::INT32;
    case proto::TensorShape_DataType_INT64:
      return DataType::INT64;
    case proto::TensorShape_DataType_UINT8:
      return DataType::UINT8;
    case proto::TensorShape_DataType_UINT16:
      return DataType::UINT16;
    case proto::TensorShape_DataType_UINT32:
      return DataType::UINT32;
    case proto::TensorShape_DataType_UINT64:
      return DataType::UINT64;
    case proto::TensorShape_DataType_FLOAT16:
      return DataType::FLOAT16;
    case proto::TensorShape_DataType_FLOAT32:
      return DataType::FLOAT32;
    case proto::TensorShape_DataType_FLOAT64:
      return DataType::FLOAT64;
    default:
      throw std::runtime_error("Unknown DataType");
  }
}

inline proto::TensorShape_DataType IntoProto(const DataType& dt) {
  switch (dt) {
    case DataType::BOOLEAN:
      return proto::TensorShape_DataType_BOOLEAN;
    case DataType::INT8:
      return proto::TensorShape_DataType_INT8;
    case DataType::INT16:
      return proto::TensorShape_DataType_INT16;
    case DataType::INT32:
      return proto::TensorShape_DataType_INT32;
    case DataType::INT64:
      return proto::TensorShape_DataType_INT64;
    case DataType::UINT8:
      return proto::TensorShape_DataType_UINT8;
    case DataType::UINT16:
      return proto::TensorShape_DataType_UINT16;
    case DataType::UINT32:
      return proto::TensorShape_DataType_UINT32;
    case DataType::UINT64:
      return proto::TensorShape_DataType_UINT64;
    case DataType::FLOAT16:
      return proto::TensorShape_DataType_FLOAT16;
    case DataType::FLOAT32:
      return proto::TensorShape_DataType_FLOAT32;
    case DataType::FLOAT64:
      return proto::TensorShape_DataType_FLOAT64;
    default:
      throw std::runtime_error("Unknown DataType");
  }
}

inline int size_in_bytes(const proto::TensorShape_DataType& dt) {
  switch (dt) {
    case proto::TensorShape_DataType_BOOLEAN:
    case proto::TensorShape_DataType_INT8:
    case proto::TensorShape_DataType_UINT8:
      return 1;
    case proto::TensorShape_DataType_INT16:
    case proto::TensorShape_DataType_UINT16:
    case proto::TensorShape_DataType_FLOAT16:
      return 2;
    case proto::TensorShape_DataType_INT32:
    case proto::TensorShape_DataType_UINT32:
    case proto::TensorShape_DataType_FLOAT32:
      return 4;
    case proto::TensorShape_DataType_INT64:
    case proto::TensorShape_DataType_UINT64:
    case proto::TensorShape_DataType_FLOAT64:
      return 8;
    default:
      throw std::runtime_error("Unknown DataType");
  }
}

inline TensorDimension FromProto(const proto::TensorShape::Dimension& dim) {
  TensorDimension ret = {dim.stride(), dim.size()};
  return ret;
}

inline proto::TensorShape::Dimension IntoProto(const TensorDimension& dim) {
  proto::TensorShape::Dimension ret;
  ret.set_size(dim.size);
  ret.set_stride(dim.stride);
  return ret;
}

inline TensorShape FromProto(const proto::TensorShape& shape) {
  TensorShape ret = {FromProto(shape.type()), {}};
  std::transform(shape.dimensions().cbegin(), shape.dimensions().cend(), std::back_inserter(ret.dims),
                 [](const proto::TensorShape::Dimension& d) { return FromProto(d); });
  return ret;
}

inline proto::TensorShape IntoProto(const TensorShape& shape) {
  proto::TensorShape ret;
  ret.set_type(IntoProto(shape.type));
  for (const auto& dim : shape.dims) {
    *(ret.mutable_dimensions()->Add()) = IntoProto(dim);
  }
  return ret;
}

inline const bool cmp(const proto::TensorShape::Dimension& a, const proto::TensorShape::Dimension& b) {
  return (a.stride() * a.size()) < (b.stride() * b.size());
}

inline int size_in_bytes(const proto::TensorShape& shape) {
  if (shape.dimensions().size() == 0) {
    return size_in_bytes(shape.type());
  }
  auto dim = *std::max_element(shape.dimensions().cbegin(), shape.dimensions().cend(), cmp);
  return dim.size() * dim.stride() * size_in_bytes(shape.type());
}

}  // namespace tile
}  // namespace vertexai

TRANSFER_ENUM(vertexai::tile::DataType);
