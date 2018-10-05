// Copyright 2017-2018 Intel Corporation.

#pragma once

#include <memory>

#include "tile/base/hal.h"

namespace vertexai {
namespace tile {
namespace local_machine {

// The platform's references to a device and the device set it came from.
struct DevInfo {
  const std::shared_ptr<hal::DeviceSet> devset;
  const std::shared_ptr<hal::Device> dev;
  const hal::proto::HardwareSettings settings;
};

}  // namespace local_machine
}  // namespace tile
}  // namespace vertexai
