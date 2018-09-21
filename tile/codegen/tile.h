// Copyright 2018, Intel Corp.

#pragma once

#include <string>
#include <vector>

#include "tile/stripe/stripe.h"

namespace vertexai {
namespace tile {
namespace codegen {

struct StencilCriteria {
  std::string name;
  int size;
  std::vector<int> out_strides;
  std::vector<int> in_strides;
};

struct StencilMatch {
  size_t total;
  std::vector<std::string> names;
  TileShape tile;
  bool is_fallback;
};

std::ostream& operator<<(std::ostream& os, const StencilMatch& match);
bool operator==(const StencilMatch& lhs, const StencilMatch& rhs);
bool operator<(const StencilMatch& lhs, const StencilMatch& rhs);

StencilMatch FindBestStencil(const std::vector<std::vector<StencilCriteria>>& criteria,  //
                             stripe::Block* block);

void ApplyTile(stripe::Block* outer, const TileShape& tile);

typedef std::function<TileShape(stripe::Block* block)> TileGenerator;

void TilePass(stripe::Block* block, const std::vector<std::vector<StencilCriteria>>& criteria);
void TilePass(stripe::Block* block, const TileGenerator& generator);

}  // namespace codegen
}  // namespace tile
}  // namespace vertexai
