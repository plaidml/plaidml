// Copyright 2017-2018 Intel Corporation.

#pragma once

#include <memory>
#include <string>
#include <unordered_map>
#include <vector>

#include "base/context/context.h"
#include "tile/base/hal.h"
#include "tile/hal/cm/cm.pb.h"
#include "tile/hal/cm/device_state.h"

namespace vertexai {
namespace tile {
namespace hal {
namespace cm {

// Device implements the hal::Device model as a single cm device.
class Device final : public hal::Device {
 public:
  Device(const context::Context& ctx, proto::DeviceInfo dinfo);

  void Initialize(const hal::proto::HardwareSettings& settings) final;

  std::string description() final;

  hal::Compiler* compiler() final { return compiler_.get(); }

  hal::Loader* loader() final { return nullptr; /* TODO: Support offline compilation */ }

  const std::unordered_map<std::string, std::unique_ptr<hal::Loader>>& il_loader_map() final { return il_loader_map_; }

  hal::Executor* executor() final { return executor_.get(); }

  std::shared_ptr<DeviceState> device_state() { return device_state_; }

 private:
  std::shared_ptr<DeviceState> device_state_;
  const std::unique_ptr<hal::Compiler> compiler_;
  const std::unordered_map<std::string, std::unique_ptr<hal::Loader>> il_loader_map_;
  const std::unique_ptr<hal::Executor> executor_;
};

}  // namespace cm
}  // namespace hal
}  // namespace tile
}  // namespace vertexai
