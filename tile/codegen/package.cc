// Copyright 2019, Intel Corp.

#include "tile/codegen/package.h"

namespace vertexai {
namespace tile {
namespace codegen {

void PackagePass::Apply(CompilerState* state) const {
  auto reqs = stripe::FromProto(options_.reqs());

  std::queue<stripe::Block*> todo;
  todo.push(state->entry());

  auto pkg_tags = stripe::FromProto(options_.package_set());
  auto subblock_tags = stripe::FromProto(options_.subblock_set());

  while (todo.size()) {
    stripe::Block* outer = todo.front();
    todo.pop();

    for (auto& stmt : outer->stmts) {
      stripe::Block* inner = dynamic_cast<stripe::Block*>(stmt.get());
      if (!inner) {
        continue;
      }

      // Add the subblock to the set of blocks to be scanned.
      todo.push(inner);

      // Test to see whether we should package this subblock.
      if (!inner->has_tags(reqs)) {
        continue;
      }

      // Package up this subblock.
      auto pkg = std::make_shared<stripe::Block>();
      pkg->name = inner->name;
      pkg->comments = inner->comments;
      pkg->stmts.emplace_back(std::move(stmt));
      pkg->set_tags(pkg_tags);

      // The location comes from the outer block -- the inner block
      // can be iterating over a sequence of locations.
      pkg->location = outer->location;

      // Propagate the current outer indices to the package block,
      // since the inner block and refinements might use them.
      for (const auto& outer_idx : outer->idxs) {
        stripe::Index pkg_idx{outer_idx.name, 1, stripe::Affine{outer_idx.name}};
        pkg_idx.set_attrs(outer_idx);
        pkg->idxs.emplace_back(std::move(pkg_idx));
      }

      // Propagate refinements used by the inner block to the package block.
      for (auto& inner_ref : inner->refs) {
        if (inner_ref.dir == stripe::RefDir::None) {
          continue;
        }
        auto outer_ref = outer->ref_by_into(inner_ref.from);
        pkg->refs.emplace(stripe::Refinement{inner_ref.dir,                                             // dir
                                             inner_ref.from,                                            // from
                                             inner_ref.from,                                            // into
                                             std::vector<stripe::Affine>(outer_ref->access.size(), 0),  // access
                                             outer_ref->interior_shape});  // interior_shape
      }

      // Set subblock tags.
      inner->add_tags(subblock_tags);

      // Replace the subblock with the package.
      stmt = std::move(pkg);
    }
  }
}

namespace {
[[gnu::unused]] char reg = []() -> char {
  CompilePassFactory<PackagePass, proto::PackagePass>::Register();
  return 0;
}();
}  // namespace
}  // namespace codegen
}  // namespace tile
}  // namespace vertexai
