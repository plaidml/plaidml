// Copyright 2017-2018 Intel Corporation.

#pragma once

#include <llvm/ExecutionEngine/ExecutionEngine.h>
#include <string>

#include "tile/lang/generate.h"

namespace vertexai {
namespace tile {
namespace hal {
namespace cpu {

class Runtime : public llvm::RuntimeDyld::SymbolResolver {
 public:
  llvm::RuntimeDyld::SymbolInfo findSymbol(const std::string&) override;
  llvm::RuntimeDyld::SymbolInfo findSymbolInLogicalDylib(const std::string&) override;
};

}  // namespace cpu
}  // namespace hal
}  // namespace tile
}  // namespace vertexai
