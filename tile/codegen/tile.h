// Copyright 2018, Intel Corporation

#pragma once

#include <string>
#include <vector>

#include <boost/optional.hpp>

#include "tile/codegen/codegen.pb.h"
#include "tile/codegen/compile_pass.h"
#include "tile/stripe/stripe.h"

namespace vertexai {
namespace tile {
namespace codegen {

struct StencilIndexMatch {
  std::string block_idx_name;
  std::string stencil_idx_name;
  uint64_t value;
};

struct StencilMatch {
  size_t cost;
  std::vector<StencilIndexMatch> idxs;
  std::vector<stripe::Refinement*> ref_ins;
  std::vector<stripe::Refinement*> ref_outs;
};

std::ostream& operator<<(std::ostream& os, const StencilMatch& match);
bool operator==(const StencilMatch& lhs, const StencilMatch& rhs);
bool operator<(const StencilMatch& lhs, const StencilMatch& rhs);

boost::optional<StencilMatch> FindBestStencil(const std::vector<proto::Stencil>& specs, stripe::Block* block);

bool ApplyTile(stripe::Block* outer,          //
               const TileShape& shape,        //
               bool elide_trivial = true,     //
               bool copy_tags = false,        //
               bool interleave = false,       //
               bool split_unaligned = false,  //
               const std::string& location_idx_tag = std::string{});

class StencilPass final : public CompilePass {
 public:
  explicit StencilPass(const proto::StencilPass& options) : options_{options} {}
  void Apply(stripe::Block* root) const final;

 private:
  proto::StencilPass options_;
};

}  // namespace codegen
}  // namespace tile
}  // namespace vertexai
