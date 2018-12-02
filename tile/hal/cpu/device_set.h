// Copyright 2017-2018 Intel Corporation.

#pragma once

#include <memory>
#include <vector>

#include "base/context/context.h"
#include "tile/base/hal.h"

namespace vertexai {
namespace tile {
namespace hal {
namespace cpu {

class DeviceSet final : public hal::DeviceSet {
 public:
  DeviceSet();

  const std::vector<std::shared_ptr<hal::Device>>& devices() final;

  hal::Memory* host_memory() final;

 private:
  std::vector<std::shared_ptr<hal::Device>> devices_;
  std::unique_ptr<hal::Memory> host_memory_;
};

}  // namespace cpu
}  // namespace hal
}  // namespace tile
}  // namespace vertexai
