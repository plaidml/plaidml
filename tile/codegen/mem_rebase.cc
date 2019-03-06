// Copyright 2018, Intel Corp.

#include "tile/codegen/mem_rebase.h"

#include <queue>

namespace vertexai {
namespace tile {
namespace codegen {

void MemRebasePass(stripe::Block* root, const proto::MemRebasePass& options) {
  auto loc = stripe::FromProto(options.loc());
  auto offset = stripe::FromProto(options.offset());

  std::queue<stripe::Block*> todo;
  todo.push(root);

  while (todo.size()) {
    stripe::Block* block = todo.front();
    todo.pop();

    for (stripe::Refinement& ref : block->refs) {
      if (ref.location.name != options.name()) {
        continue;
      }
      std::map<std::string, std::int64_t> vars{{"unit", ref.location.unit.constant()}, {"offset", ref.offset}};
      ref.location.name = loc.name;
      ref.location.unit = loc.unit.partial_eval(vars);
      ref.offset = offset.eval(vars);
    }

    for (auto& stmt : block->stmts) {
      stripe::Block* inner = dynamic_cast<stripe::Block*>(stmt.get());
      if (inner) {
        todo.push(inner);
      }
    }
  }
}

}  // namespace codegen
}  // namespace tile
}  // namespace vertexai
