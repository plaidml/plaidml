// Copyright 2017-2018 Intel Corporation.

#pragma once

#include <memory>
#include <string>
#include <vector>

#include "tile/base/hal.h"
#include "tile/lang/generate.h"

namespace llvm {
class ExecutionEngine;
}

namespace vertexai {
namespace tile {
namespace hal {
namespace cpu {

class Library final : public hal::Library {
 public:
  static Library* Downcast(hal::Library* library);

  Library(const std::vector<std::shared_ptr<llvm::ExecutionEngine>>& engines,
          const std::vector<lang::KernelInfo>& kernels);

  std::string Serialize() final { return ""; }

  const std::vector<std::shared_ptr<llvm::ExecutionEngine>>& engines() { return engines_; }
  const std::vector<lang::KernelInfo>& kernels() { return kernels_; }

 private:
  std::vector<std::shared_ptr<llvm::ExecutionEngine>> engines_;
  std::vector<lang::KernelInfo> kernels_;
};

}  // namespace cpu
}  // namespace hal
}  // namespace tile
}  // namespace vertexai
