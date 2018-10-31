// Copyright 2018, Intel Corp.

#pragma once

#include <string>
#include <vector>

#include <boost/optional.hpp>

#include "tile/codegen/tags.h"
#include "tile/stripe/stripe.h"

namespace vertexai {
namespace tile {
namespace codegen {

struct StencilIndex {
  std::string name;
  int size;
  std::vector<int> out_strides;
  std::vector<int> in_strides;
};

struct StencilSpec {
  std::string name;
  size_t startup_cost;
  std::vector<StencilIndex> idxs;
};

struct StencilIndexMatch {
  std::string block_idx_name;
  std::string stencil_idx_name;
  uint64_t value;
};

struct StencilMatch {
  std::string name;
  size_t cost;
  std::vector<StencilIndexMatch> idxs;
  bool is_fallback;
};

std::ostream& operator<<(std::ostream& os, const StencilMatch& match);
bool operator==(const StencilMatch& lhs, const StencilMatch& rhs);
bool operator<(const StencilMatch& lhs, const StencilMatch& rhs);

boost::optional<StencilMatch> FindBestStencil(const std::vector<StencilSpec>& specs, const stripe::Block& block);

void ApplyTile(stripe::Block* inner, const TileShape& shape, bool elide_trivial = true);

struct StencilPassOptions {
  Tags reqs;
  std::vector<StencilSpec> specs;
  Tags set_outer;
  Tags set_inner;
};

void StencilPass(stripe::Block* block, const StencilPassOptions& options);

}  // namespace codegen
}  // namespace tile
}  // namespace vertexai
