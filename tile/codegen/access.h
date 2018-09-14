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
};

struct Constraint {
  std::vector<int64_t> lhs;
  int64_t rhs;
};

struct AccessPattern {
  bool is_write = false;
  bool is_exact = false;
  int64_t offset = 0;
  std::vector<IndexAccess> access;
  std::vector<Constraint> constraints;
};

std::vector<AccessPattern> ComputeAccess(const stripe::proto::Block& block, const std::string& buffer);

}  // namespace codegen
}  // namespace tile
}  // namespace vertexai
