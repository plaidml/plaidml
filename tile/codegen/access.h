// Copyright 2018, Intel Corp.

#pragma once

#include "tile/lang/generate.h"
#include "tile/proto/stripe.pb.h"

namespace vertexai {
namespace tile {
namespace codegen {

struct IndexAccess {
  std::string name;
  int64_t stride;
  uint64_t range;
  bool operator==(const IndexAccess& o) const { 
    return name == o.name && stride == o.stride && range == o.range;
  }
};

struct Constraint {
  std::vector<int64_t> lhs;
  int64_t rhs;
  bool operator==(const Constraint& o) const { 
    return lhs == o.lhs && rhs == o.rhs;
  }
};

struct AccessPattern {
  bool is_write = false;
  bool is_exact = false;
  int64_t offset = 0;
  std::vector<IndexAccess> access;
  std::vector<Constraint> constraints;
  bool operator==(const AccessPattern& o) const { 
    return is_write == o.is_write && is_exact == o.is_exact && offset == o.offset && access == o.access && constraints == o.constraints;
  }
};

std::ostream& operator<<(std::ostream& stream, const AccessPattern& ap);

std::vector<AccessPattern> ComputeAccess(const stripe::proto::Block& block, const std::string& buffer);

}  // namespace codegen
}  // namespace tile
}  // namespace vertexai
