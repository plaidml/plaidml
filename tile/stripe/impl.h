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

struct Taggable::Impl {
  std::map<std::string, AttrValue> attrs;
};

struct Accessor {
  static const Taggable::Impl* impl(const Taggable& taggable);
};

}  // namespace stripe
}  // namespace tile
}  // namespace vertexai
