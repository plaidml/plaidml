// Copyright 2020, Intel Corporation

#include "pmlc/rt/runtime_registry.h"

#include <string>
#include <unordered_map>

namespace pmlc::rt {

std::unordered_map<std::string, std::unique_ptr<Runtime>> &getRuntimeMap() {
  static std::unordered_map<std::string, std::unique_ptr<Runtime>> runtimeMap;
  return runtimeMap;
}

void registerRuntime(llvm::StringRef id, std::unique_ptr<Runtime> runtime) {
  getRuntimeMap()[id.str()] = std::move(runtime);
}

} // namespace pmlc::rt
