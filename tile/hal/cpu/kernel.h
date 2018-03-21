// Copyright 2017, Vertex.AI. CONFIDENTIAL

#pragma once

#include <array>
#include <memory>
#include <string>
#include <vector>

#include "tile/base/hal.h"

namespace llvm {
  class ExecutionEngine;
}

namespace vertexai {
namespace tile {
namespace hal {
namespace cpu {

class Kernel final : public hal::Kernel {
 public:
  Kernel(std::shared_ptr<llvm::ExecutionEngine> engine, const lang::KernelInfo& ki);

  std::shared_ptr<hal::Event> Run(const context::Context& ctx, const std::vector<std::shared_ptr<hal::Buffer>>& params,
                                  const std::vector<std::shared_ptr<hal::Event>>& dependencies,
                                  bool enable_profiling) final;

  static std::string InvokerName(std::string kname);

 private:
  std::shared_ptr<llvm::ExecutionEngine> engine_;
  lang::KernelInfo ki_;
  static const char invoker_prefix_[];
};

}  // namespace cpu
}  // namespace hal
}  // namespace tile
}  // namespace vertexai
