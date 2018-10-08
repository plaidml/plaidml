// Copyright 2017-2018 Intel Corporation.

#pragma once

#include <memory>
#include <vector>

#include "tile/base/hal.h"

namespace llvm {
class ExecutionEngine;
class Module;
}  // namespace llvm

namespace vertexai {
namespace tile {
namespace hal {
namespace cpu {

class Compiler final : public hal::Compiler {
 public:
  Compiler();

  boost::future<std::unique_ptr<hal::Library>> Build(const context::Context& ctx,
                                                     const std::vector<lang::KernelInfo>& kernels,
                                                     const hal::proto::HardwareSettings& /* settings */) final;

 private:
  void BuildKernel(const lang::KernelInfo&, std::vector<std::shared_ptr<llvm::ExecutionEngine>>* engines);
  void GenerateInvoker(const lang::KernelInfo&, llvm::Module*);
};

}  // namespace cpu
}  // namespace hal
}  // namespace tile
}  // namespace vertexai
