// Copyright 2018, Intel Corp.

#pragma once

#include <string>
#include <vector>

#include "tile/stripe/stripe.h"

namespace vertexai {
namespace tile {
namespace codegen {

struct AccessPattern {
  bool is_write;
  bool is_exact;
  std::vector<stripe::Index> idxs;
  stripe::BufferAccess access;
  std::vector<stripe::Constraint> constraints;
};

bool operator==(const AccessPattern& lhs, const AccessPattern& rhs);

std::ostream& operator<<(std::ostream& os, const AccessPattern& ap);

std::vector<AccessPattern> ComputeAccess(const stripe::Block& block, const std::string& buffer);

}  // namespace codegen
}  // namespace tile
}  // namespace vertexai
