// Copyright 2018, Intel Corp.

#pragma once

#include <string>
#include <vector>

#include "tile/lang/generate.h"
#include "tile/proto/stripe.pb.h"

namespace vertexai {
namespace tile {
namespace codegen {

struct BufferAccess {
  int64_t offset;
  std::vector<int64_t> strides;
  bool operator==(const BufferAccess& o) const { return offset == o.offset && strides == o.strides; }
};

std::ostream& operator<<(std::ostream& stream, const BufferAccess& a);

struct Index {
  std::string name;
  uint64_t range;
  int64_t factor;
  bool operator==(const Index& o) const { return name == o.name && range == o.range && factor == o.factor; }
};
typedef std::vector<Index> Indexes;

std::ostream& operator<<(std::ostream& stream, const Index& a);

struct Constraint {
  std::vector<int64_t> lhs;
  int64_t rhs;
  bool operator==(const Constraint& o) const { return lhs == o.lhs && rhs == o.rhs; }
};
typedef std::vector<Constraint> Constraints;

struct AccessPattern {
  bool is_write;
  bool is_exact;
  Indexes indexes;
  BufferAccess access;
  Constraints constraints;
  bool operator==(const AccessPattern& o) const {
    return is_write == o.is_write && is_exact == o.is_exact && indexes == o.indexes && access == o.access &&
           constraints == o.constraints;
  }
};

std::ostream& operator<<(std::ostream& stream, const AccessPattern& ap);

std::vector<AccessPattern> ComputeAccess(const stripe::proto::Block& block, const std::string& buffer);

}  // namespace codegen
}  // namespace tile
}  // namespace vertexai
