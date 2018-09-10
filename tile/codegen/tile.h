// Copyright 2018, Intel Corp.

#pragma once

#include "tile/lang/generate.h"
#include "tile/proto/stripe.pb.h"

namespace vertexai {
namespace tile {
namespace codegen {

struct StencilCriteria {
  std::string name;
  int size;
  std::vector<int> strides;
};

struct StencilMatch {
  size_t total;
  std::vector<std::string> names;
  lang::TileShape tile;
};

MAKE_LOGGABLE(StencilMatch, match, os);
bool operator==(const StencilMatch& lhs, const StencilMatch& rhs);
bool operator<(const StencilMatch& lhs, const StencilMatch& rhs);

StencilMatch FindBestStencil(const std::vector<std::vector<StencilCriteria>>& criteria,  //
                             const stripe::proto::Block& block);

void ApplyTile(stripe::proto::Block* outer, const lang::TileShape& tile);

typedef std::function<lang::TileShape(const stripe::proto::Block& block)> TileGenerator;

void TilePass(stripe::proto::Block* block, const std::vector<std::vector<StencilCriteria>>& criteria);
void TilePass(stripe::proto::Block* block, const TileGenerator& generator);

}  // namespace codegen
}  // namespace tile
}  // namespace vertexai
