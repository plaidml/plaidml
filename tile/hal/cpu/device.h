// Copyright 2017, Vertex.AI. CONFIDENTIAL

#pragma once

#include <memory>
#include <string>
#include <unordered_map>
#include <vector>

#include "base/context/context.h"
#include "tile/base/hal.h"
#include "tile/hal/cpu/cpu.pb.h"

namespace vertexai {
namespace tile {
namespace hal {
namespace cpu {

// Device implements the hal::Device model as a single OpenCL device.
class Device final : public hal::Device {
 public:
  Device();

  void Initialize(const hal::proto::HardwareSettings& settings) final {
    // NOP
  }

  std::string description() final { return "LLVM_preview_CPU"; }

  hal::Compiler* compiler() final { return compiler_.get(); }

  hal::Loader* loader() final { return loader_.get(); }

  const std::unordered_map<std::string, std::unique_ptr<hal::Loader>>& il_loader_map() final { return il_loader_map_; }

  hal::Executor* executor() final { return executor_.get(); }

 private:
  const std::unique_ptr<hal::Compiler> compiler_;
  const std::unique_ptr<hal::Loader> loader_;
  const std::unordered_map<std::string, std::unique_ptr<hal::Loader>> il_loader_map_;
  const std::unique_ptr<hal::Executor> executor_;
};

}  // namespace cpu
}  // namespace hal
}  // namespace tile
}  // namespace vertexai
