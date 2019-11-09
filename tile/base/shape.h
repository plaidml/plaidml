#pragma once

#include <algorithm>
#include <map>
#include <ostream>
#include <set>
#include <string>
#include <utility>
#include <vector>

#include "base/util/transfer_object.h"
#include "tile/proto/shape.pb.h"

namespace vertexai {
namespace tile {

typedef std::vector<uint64_t> TileShape;

inline int64_t tile_shape_product(const TileShape& shape) {
  int64_t product = 1;
  for (auto dim : shape) {
    product *= dim;
  }
  return product;
}

enum class DataType : int {
  INVALID = 0,
  BOOLEAN = 0x02,
  INT8 = 0x10,
  INT16 = 0x11,
  INT32 = 0x12,
  INT64 = 0x13,
  INT128 = 0x14,
  INTX = 0x1F,
  UINT8 = 0x20,
  UINT16 = 0x21,
  UINT32 = 0x22,
  UINT64 = 0x23,
  UINTX = 0x2F,
  FLOAT16 = 0x31,
  FLOAT32 = 0x32,
  FLOAT64 = 0x33,
  FLOATX = 0x3F,
  BFLOAT16 = 0x38,
  PRNG = 0x40,
};

inline const std::set<DataType>& GetDataTypeSet() {
  static std::set<DataType> all_types = {
      DataType::BOOLEAN,   //
      DataType::INT8,      //
      DataType::INT16,     //
      DataType::INT32,     //
      DataType::INT64,     //
      DataType::INT128,    //
      DataType::INTX,      //
      DataType::UINT8,     //
      DataType::UINT16,    //
      DataType::UINT32,    //
      DataType::UINT64,    //
      DataType::UINTX,     //
      DataType::FLOAT16,   //
      DataType::FLOAT32,   //
      DataType::FLOAT64,   //
      DataType::FLOATX,    //
      DataType::BFLOAT16,  //
  };
  return all_types;
}

inline bool is_int(const DataType& dt) {
  switch (dt) {
    case DataType::INT8:
    case DataType::INT16:
    case DataType::INT32:
    case DataType::INT64:
    case DataType::INT128:
    case DataType::INTX:
      return true;
    default:
      return false;
  }
}

inline bool is_uint(const DataType& dt) {
  switch (dt) {
    case DataType::UINT8:
    case DataType::UINT16:
    case DataType::UINT32:
    case DataType::UINT64:
    case DataType::UINTX:
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
    case DataType::FLOATX:
    case DataType::BFLOAT16:
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
    case DataType::BFLOAT16:
      return 16;
    case DataType::INT128:
      return 128;
    default:
      return 0;
  }
}

inline std::string to_string(const DataType& dt) {
  switch (dt) {
    case DataType::INVALID:
      return "void";
    case DataType::BOOLEAN:
      return "bool";
    case DataType::INT8:
      return "i8";
    case DataType::INT16:
      return "i16";
    case DataType::INT32:
      return "i32";
    case DataType::INT64:
      return "i64";
    case DataType::INT128:
      return "i128";
    case DataType::INTX:
      return "int";
    case DataType::UINT8:
      return "u8";
    case DataType::UINT16:
      return "u16";
    case DataType::UINT32:
      return "u32";
    case DataType::UINT64:
      return "u64";
    case DataType::UINTX:
      return "uint";
    case DataType::FLOAT16:
      return "fp16";
    case DataType::FLOAT32:
      return "fp32";
    case DataType::FLOAT64:
      return "fp64";
    case DataType::FLOATX:
      return "float";
    case DataType::BFLOAT16:
      return "bf16";
    case DataType::PRNG:
      return "prng";
    default:
      return "!!invalid data type: " + std::to_string(static_cast<int>(dt));
  }
}

inline DataType DataTypeFromString(const std::string& str) {
  static std::map<std::string, DataType> tbl = {
      {"void", DataType::INVALID},   //
      {"bool", DataType::BOOLEAN},   //
      {"i8", DataType::INT8},        //
      {"i16", DataType::INT16},      //
      {"i32", DataType::INT32},      //
      {"i64", DataType::INT64},      //
      {"i128", DataType::INT128},    //
      {"int", DataType::INTX},       //
      {"u8", DataType::UINT8},       //
      {"u16", DataType::UINT16},     //
      {"u32", DataType::UINT32},     //
      {"u64", DataType::UINT64},     //
      {"uint", DataType::UINTX},     //
      {"fp16", DataType::FLOAT16},   //
      {"fp32", DataType::FLOAT32},   //
      {"fp64", DataType::FLOAT64},   //
      {"float", DataType::FLOATX},   //
      {"bf16", DataType::BFLOAT16},  //
      {"prng", DataType::PRNG},      //
  };
  auto it = tbl.find(str);
  if (it == tbl.end()) {
    throw std::runtime_error("Unknown datatype: " + str);
  }
  return it->second;
}

inline size_t byte_width(const DataType& dt) { return (bit_width(dt) + 7) / 8; }

// Compute result type by 'upcasting' to the highest type in the hierarchy
DataType CommonSupertype(DataType left, DataType right);

struct TensorDimension {
  TensorDimension() = default;
  TensorDimension(int64_t stride, uint64_t size) : stride(stride), size(size) {}

  int64_t stride;  // Stride over element count
  uint64_t size;   // Number of elements

  inline bool operator==(const TensorDimension& rhs) const {
    return std::tie(stride, size) ==  //
           std::tie(rhs.stride, rhs.size);
  }

  inline bool operator<(const TensorDimension& rhs) const {
    return std::tie(stride, size) <  //
           std::tie(rhs.stride, rhs.size);
  }
};

struct TensorShape {
  TensorShape() = default;
  TensorShape(DataType type, const std::vector<TensorDimension>& dims, const std::string& layout = "")
      : type(type), dims(dims), layout(layout) {}

  DataType type = DataType::INVALID;
  std::vector<TensorDimension> dims;
  bool is_const = false;
  std::string codec;
  std::string layout;

  uint64_t byte_size() const { return elem_size() * byte_width(type); }

  uint64_t elem_size() const {
    uint64_t max_elem = 0;
    for (const auto& dim : dims) {
      if (!dim.size) {
        return 0;
      }
      if (dim.stride > 0) {
        max_elem += (dim.size - 1) * dim.stride;
      }
    }
    return max_elem + 1;
  }

  std::vector<size_t> sizes() const {
    std::vector<size_t> ret;
    for (const auto& dim : dims) {
      ret.push_back(dim.size);
    }
    return ret;
  }

  std::vector<size_t> strides() const {
    std::vector<size_t> ret;
    for (const auto& dim : dims) {
      ret.push_back(dim.stride);
    }
    return ret;
  }

  size_t sizes_product() const {
    size_t ret = 1;
    for (const auto& size : sizes()) {
      ret *= size;
    }
    return ret;
  }

  size_t sizes_product_bytes() const { return sizes_product() * byte_width(type); }

  // Sort dims from low stride to high stride
  std::vector<TensorDimension> natural_dims() const {
    std::vector<TensorDimension> ret = dims;
    std::sort(ret.begin(), ret.end(), [](const TensorDimension& a, const TensorDimension& b) {
      return std::abs(a.stride) < std::abs(b.stride);
    });
    return ret;
  }

  // Expected number of cache lines hit given random alignment
  double memory_io(size_t cache_width) const {
    double cache_elems = static_cast<double>(cache_width) / byte_width(type);
    // Start with one cache line
    double cache_lines = 1.0;
    // Current accumulated maximum value
    int64_t max_val = 0;
    // For each dimension (in sorted order)
    for (const auto& dim : natural_dims()) {
      // Compute gap per step
      int64_t gap = std::abs(dim.stride) - max_val;
      // Multiply current cache hits by size
      cache_lines *= static_cast<double>(dim.size);
      // Compute probability that cache line is shared across gap
      double prob_shared = 0.0;  // Assume it's never shared
      if (cache_elems != 0.0 && gap < cache_elems) {
        prob_shared = 1.0 - (gap / cache_elems);
      }
      // Subtract shared elements
      cache_lines -= prob_shared * static_cast<double>(dim.size - 1);
      // Update max_val
      max_val += std::abs(dim.stride) * (dim.size - 1);
    }
    return cache_lines;
  }

  inline bool operator==(const TensorShape& rhs) const {
    return std::tie(type, dims) ==  //
           std::tie(rhs.type, rhs.dims);
  }

  inline bool operator<(const TensorShape& rhs) const {
    return std::tie(type, dims) <  //
           std::tie(rhs.type, rhs.dims);
  }

  void resize_dim(size_t pos, uint64_t size);
};

std::ostream& operator<<(std::ostream& os, const TensorShape& shape);
std::ostream& operator<<(std::ostream& os, const TensorDimension& dim);

inline TensorShape SimpleShape(DataType type, const std::vector<size_t>& sizes, const std::string& layout = "") {
  int64_t stride = 1;
  std::vector<TensorDimension> dims(sizes.size());
  for (int i = sizes.size() - 1; i >= 0; i--) {
    dims[i].stride = stride;
    dims[i].size = sizes[i];
    stride *= sizes[i];
  }
  return TensorShape(type, dims, layout);
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
    case proto::TensorShape_DataType_INT128:
      return DataType::INT128;
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
    case proto::TensorShape_DataType_BFLOAT16:
      return DataType::BFLOAT16;
    case proto::TensorShape_DataType_PRNG:
      return DataType::PRNG;
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
    case DataType::INT128:
      return proto::TensorShape_DataType_INT128;
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
    case DataType::BFLOAT16:
      return proto::TensorShape_DataType_BFLOAT16;
    case DataType::PRNG:
      return proto::TensorShape_DataType_PRNG;
    default:
      throw std::runtime_error("Unknown DataType");
  }
}

inline TensorDimension FromProto(const proto::TensorShape::Dimension& dim) {  //
  return {dim.stride(), dim.size()};
}

inline proto::TensorShape::Dimension IntoProto(const TensorDimension& dim) {
  proto::TensorShape::Dimension ret;
  ret.set_size(dim.size);
  ret.set_stride(dim.stride);
  return ret;
}

inline TensorShape FromProto(const proto::TensorShape& shape) {
  TensorShape ret;
  ret.type = FromProto(shape.type());
  ret.codec = shape.codec();
  ret.is_const = shape.is_const();
  ret.layout = shape.layout();
  for (const auto& dim : shape.dims()) {
    ret.dims.emplace_back(FromProto(dim));
  }
  return ret;
}

inline proto::TensorShape IntoProto(const TensorShape& shape) {
  proto::TensorShape ret;
  ret.set_type(IntoProto(shape.type));
  ret.set_codec(shape.codec);
  ret.set_is_const(shape.is_const);
  if (!shape.layout.empty()) {
    ret.set_layout(shape.layout);
  }
  for (const auto& dim : shape.dims) {
    *(ret.mutable_dims()->Add()) = IntoProto(dim);
  }
  return ret;
}

}  // namespace tile
}  // namespace vertexai

TRANSFER_ENUM(vertexai::tile::DataType);
