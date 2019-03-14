// Copyright 2017-2018 Intel Corporation.

#pragma once

#include <array>
#include <memory>
#include <string>
#include <vector>

#include <boost/asio/thread_pool.hpp>

#include "tile/base/hal.h"

namespace llvm {
class ExecutionEngine;
class LLVMContext;
}  // namespace llvm

namespace vertexai {
namespace tile {
namespace hal {
namespace cpu {

class Executable final : public hal::Executable {
 public:
  Executable(std::shared_ptr<llvm::LLVMContext> llvm_context,
             std::vector<std::shared_ptr<llvm::ExecutionEngine>> engines, std::vector<lang::KernelInfo> kis,
             std::shared_ptr<boost::asio::thread_pool> thread_pool);
  virtual ~Executable();

  std::shared_ptr<hal::Event> Run(const context::Context& ctx, std::size_t kidx,
                                  const std::vector<std::shared_ptr<hal::Buffer>>& params,
                                  const std::vector<std::shared_ptr<hal::Event>>& dependencies,
                                  bool enable_profiling) final;

  static std::string InvokerName(std::string kname);

 private:
  std::shared_ptr<llvm::LLVMContext> llvm_context_;
  std::vector<std::shared_ptr<llvm::ExecutionEngine>> engines_;
  std::vector<lang::KernelInfo> kis_;
  std::shared_ptr<boost::asio::thread_pool> thread_pool_;
};

}  // namespace cpu
}  // namespace hal
}  // namespace tile
}  // namespace vertexai
