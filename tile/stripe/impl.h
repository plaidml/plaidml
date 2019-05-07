// Copyright 2018, Intel Corporation

#pragma once

#include <map>
#include <string>

#include <boost/variant.hpp>

#include "tile/stripe/stripe.h"

namespace vertexai {
namespace tile {
namespace stripe {

struct Void {};

inline std::ostream& operator<<(std::ostream& os, const Void& value) {
  os << "{}";
  return os;
}

using AttrValue = boost::variant<  //
    Void,                          //
    bool,                          //
    int64_t,                       //
    double,                        //
    std::string,                   //
    google::protobuf::Any>;

class AttrValueToStringVisitor : public boost::static_visitor<std::string> {
 public:
  std::string operator()(const Void& v) const { return "{}"; }
  std::string operator()(const bool& v) const { return std::to_string(v); }
  std::string operator()(const int64_t& v) const { return std::to_string(v); }
  std::string operator()(const double& v) const { return std::to_string(v); }
  std::string operator()(const std::string& v) const { return v; }
  std::string operator()(const google::protobuf::Any& v) const { return "<protobuf::Any object>"; }
};

inline std::string to_string(AttrValue val) { return boost::apply_visitor(AttrValueToStringVisitor(), val); }

inline std::ostream& operator<<(std::ostream& os, const AttrValue& value) {
  os << to_string(value);
  return os;
}

struct Taggable::Impl {
  std::map<std::string, AttrValue> attrs;
};

struct Accessor {
  static const Taggable::Impl* impl(const Taggable& taggable);
};

}  // namespace stripe
}  // namespace tile
}  // namespace vertexai
