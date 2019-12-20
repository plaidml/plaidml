// Copyright 2018, Intel Corp.

#include "tile/codegen/rewrite_locs.h"

#include <algorithm>
#include <iterator>
#include <queue>
#include <utility>
#include <vector>

namespace vertexai {
namespace tile {
namespace codegen {
namespace {

struct Rewrite {
  std::vector<stripe::Device> prefix;
  std::vector<stripe::Device> target;
};

void RewriteLocation(stripe::Location* loc, const std::vector<Rewrite>& rewrites) {
  for (auto& rewrite : rewrites) {
    auto loc_it = loc->devs.begin();
    auto rew_it = rewrite.prefix.begin();
    for (; loc_it != loc->devs.end() && rew_it != rewrite.prefix.end() && *loc_it == *rew_it; ++loc_it, ++rew_it) {
    }
    if (rew_it == rewrite.prefix.end()) {
      std::vector<stripe::Device> target = rewrite.target;
      std::copy(loc_it, loc->devs.end(), std::back_inserter(target));
      std::swap(loc->devs, target);
      break;
    }
  }
}

}  // namespace

void RewriteLocationsPass::Apply(CompilerState* state) const {
  std::vector<Rewrite> rewrites;
  for (const auto& rewrite : options_.rewrites()) {
    rewrites.emplace_back(Rewrite{stripe::FromProto(rewrite.prefix()), stripe::FromProto(rewrite.target())});
  }

  std::queue<stripe::Block*> todo;
  todo.push(state->entry());

  while (todo.size()) {
    stripe::Block* block = todo.front();
    todo.pop();

    RewriteLocation(&block->location, rewrites);

    for (auto& ref : block->refs) {
      RewriteLocation(&ref.mut().location, rewrites);
    }

    for (auto& stmt : block->stmts) {
      stripe::Block* inner = dynamic_cast<stripe::Block*>(stmt.get());
      if (inner) {
        todo.push(inner);
      }
    }
  }
}

namespace {
[[gnu::unused]] char reg = []() -> char {
  CompilePassFactory<RewriteLocationsPass, proto::RewriteLocationsPass>::Register();
  return 0;
}();
}  // namespace
}  // namespace codegen
}  // namespace tile
}  // namespace vertexai
