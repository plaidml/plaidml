// Copyright 2018, Intel Corp.

#include "tile/codegen/mem_rebase.h"

#include <queue>

namespace vertexai {
namespace tile {
namespace codegen {

void MemRebasePass(stripe::Block* root, const proto::MemRebasePass& options) {
  auto offset = stripe::FromProto(options.offset());

  std::queue<stripe::Block*> todo;
  todo.push(root);

  while (todo.size()) {
    stripe::Block* block = todo.front();
    todo.pop();

    for (auto& ref : block->refs) {
      if (ref.location != options.pattern()) {
        continue;
      }
      std::map<std::string, std::int64_t> vars{{"offset", ref.offset}};
      for (const auto& dev : ref.location.devs) {
        for (std::size_t idx = 0; idx < dev.units.size(); ++idx) {
          vars[dev.name + '.' + std::to_string(idx)] = dev.units[idx].constant();
        }
      }
      ref.mut().offset = offset.eval(vars);
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
