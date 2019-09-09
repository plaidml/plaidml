// Copyright 2017-2018 Intel Corporation.

#pragma once

#include <memory>
#include <vector>

#include "tile/base/hal.h"
#include "tile/hal/cm/device_state.h"

namespace vertexai {
namespace tile {
namespace hal {
namespace cm {

class Compiler final : public hal::Compiler {
 public:
  explicit Compiler(std::shared_ptr<DeviceState> device_state);

  boost::future<std::unique_ptr<hal::Library>> Build(const context::Context& ctx,
                                                     const std::vector<lang::KernelInfo>& kernels,
                                                     const hal::proto::HardwareSettings& settings) final;

  static int knum;

 private:
  std::shared_ptr<DeviceState> device_state_;
};

}  // namespace cm
}  // namespace hal
}  // namespace tile
}  // namespace vertexai
