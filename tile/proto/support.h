#pragma once

#include <algorithm>
#include <iterator>
#include <map>
#include <string>
#include <utility>

#include "tile/base/shape.h"
#include "tile/proto/tile.pb.h"

namespace vertexai {
namespace tile {

inline ShapeMap FromProto(const google::protobuf::Map<std::string, proto::ProgramInput>& pb_map) {
  std::map<std::string, TensorShape> ret;
  std::for_each(pb_map.cbegin(), pb_map.cend(), [&ret](const std::pair<std::string, proto::ProgramInput>& item) {
    ret[item.first] = FromProto(item.second.shape());
  });
  return ret;
}

inline ShapeMap FromProto(const google::protobuf::Map<std::string, proto::ProgramOutput>& pb_map) {
  std::map<std::string, TensorShape> ret;
  std::for_each(pb_map.cbegin(), pb_map.cend(), [&ret](const std::pair<std::string, proto::ProgramOutput>& item) {
    ret[item.first] = FromProto(item.second.shape());
  });
  return ret;
}

}  // namespace tile
}  // namespace vertexai
