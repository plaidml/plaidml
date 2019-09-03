// Copyright 2019, Intel Corp.

#pragma once

#include <llvm/IR/Module.h>

#include <map>
#include <memory>
#include <string>
#include <vector>

namespace vertexai {
namespace tile {
namespace targets {
namespace cpu {

struct ProgramModule {
  std::unique_ptr<llvm::Module> module;
  std::vector<std::string> parameters;
  std::map<std::string, void*> externals;
};

}  // namespace cpu
}  // namespace targets
}  // namespace tile
}  // namespace vertexai
