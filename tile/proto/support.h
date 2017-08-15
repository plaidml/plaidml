#pragma once

#include <algorithm>
#include <iterator>
#include <string>
#include <utility>

#include "base/util/logging.h"
#include "tile/lang/shape.h"
#include "tile/proto/tile.pb.h"

namespace vertexai {
namespace tile {
namespace proto {

inline lang::DataType to_poco(const TensorShape_DataType& dt) {
  switch (dt) {
    case TensorShape_DataType_BOOLEAN:
      return lang::DataType::BOOLEAN;
    case TensorShape_DataType_INT8:
      return lang::DataType::INT8;
    case TensorShape_DataType_INT16:
      return lang::DataType::INT16;
    case TensorShape_DataType_INT32:
      return lang::DataType::INT32;
    case TensorShape_DataType_INT64:
      return lang::DataType::INT64;
    case TensorShape_DataType_UINT8:
      return lang::DataType::UINT8;
    case TensorShape_DataType_UINT16:
      return lang::DataType::UINT16;
    case TensorShape_DataType_UINT32:
      return lang::DataType::UINT32;
    case TensorShape_DataType_UINT64:
      return lang::DataType::UINT64;
    case TensorShape_DataType_FLOAT16:
      return lang::DataType::FLOAT16;
    case TensorShape_DataType_FLOAT32:
      return lang::DataType::FLOAT32;
    case TensorShape_DataType_FLOAT64:
      return lang::DataType::FLOAT64;
    default:
      throw std::runtime_error("Unknown DataType");
  }
}

inline TensorShape_DataType to_proto(const lang::DataType& dt) {
  switch (dt) {
    case lang::DataType::BOOLEAN:
      return TensorShape_DataType_BOOLEAN;
    case lang::DataType::INT8:
      return TensorShape_DataType_INT8;
    case lang::DataType::INT16:
      return TensorShape_DataType_INT16;
    case lang::DataType::INT32:
      return TensorShape_DataType_INT32;
    case lang::DataType::INT64:
      return TensorShape_DataType_INT64;
    case lang::DataType::UINT8:
      return TensorShape_DataType_UINT8;
    case lang::DataType::UINT16:
      return TensorShape_DataType_UINT16;
    case lang::DataType::UINT32:
      return TensorShape_DataType_UINT32;
    case lang::DataType::UINT64:
      return TensorShape_DataType_UINT64;
    case lang::DataType::FLOAT16:
      return TensorShape_DataType_FLOAT16;
    case lang::DataType::FLOAT32:
      return TensorShape_DataType_FLOAT32;
    case lang::DataType::FLOAT64:
      return TensorShape_DataType_FLOAT64;
    default:
      throw std::runtime_error("Unknown DataType");
  }
}

inline int size_in_bytes(const TensorShape_DataType& dt) {
  switch (dt) {
    case TensorShape_DataType_BOOLEAN:
    case TensorShape_DataType_INT8:
    case TensorShape_DataType_UINT8:
      return 1;
    case TensorShape_DataType_INT16:
    case TensorShape_DataType_UINT16:
    case TensorShape_DataType_FLOAT16:
      return 2;
    case TensorShape_DataType_INT32:
    case TensorShape_DataType_UINT32:
    case TensorShape_DataType_FLOAT32:
      return 4;
    case TensorShape_DataType_INT64:
    case TensorShape_DataType_UINT64:
    case TensorShape_DataType_FLOAT64:
      return 8;
    default:
      throw std::runtime_error("Unknown DataType");
  }
}

inline lang::TensorDimension to_poco(const TensorShape::Dimension& dim) {
  lang::TensorDimension ret = {dim.stride(), dim.size()};
  return ret;
}

inline TensorShape::Dimension to_proto(const lang::TensorDimension& dim) {
  TensorShape::Dimension ret;
  ret.set_size(dim.size);
  ret.set_stride(dim.stride);
  return ret;
}

inline lang::TensorShape to_poco(const TensorShape& shape) {
  lang::TensorShape ret = {to_poco(shape.type()), {}};
  std::transform(shape.dimensions().cbegin(), shape.dimensions().cend(), std::back_inserter(ret.dims),
                 [](const TensorShape::Dimension& d) { return to_poco(d); });
  return ret;
}

inline TensorShape to_proto(const lang::TensorShape& shape) {
  TensorShape ret;
  ret.set_type(to_proto(shape.type));
  for (const auto& dim : shape.dims) {
    *(ret.mutable_dimensions()->Add()) = to_proto(dim);
  }
  return ret;
}

inline lang::ShapeMap to_poco(const google::protobuf::Map<std::string, TensorShape> m) {
  std::map<std::string, lang::TensorShape> ret;
  std::for_each(m.cbegin(), m.cend(),
                [&ret](const std::pair<std::string, TensorShape>& p) { ret[p.first] = to_poco(p.second); });
  return ret;
};

inline google::protobuf::Map<std::string, TensorShape> to_proto(const lang::ShapeMap& m) {
  google::protobuf::Map<std::string, TensorShape> r;
  for (const auto& kvp : m) {
    r[kvp.first] = to_proto(kvp.second);
  }
  return r;
}

inline const bool cmp(const TensorShape::Dimension& a, const TensorShape::Dimension& b) {
  return (a.stride() * a.size()) < (b.stride() * b.size());
}

inline int size_in_bytes(const tile::proto::TensorShape& shape) {
  if (shape.dimensions().size() == 0) {
    return size_in_bytes(shape.type());
  }
  TensorShape::Dimension d = *std::max_element(shape.dimensions().cbegin(), shape.dimensions().cend(), cmp);
  return d.size() * d.stride() * size_in_bytes(shape.type());
}

}  // namespace proto
}  // namespace tile
}  // namespace vertexai
