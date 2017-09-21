// Copyright 2017, Vertex.AI. CONFIDENTIAL

#pragma once

#include <memory>

#include "tile/base/hal.h"
#include "tile/hal/opencl/device_state.h"
#include "tile/hal/opencl/ocl.h"

namespace vertexai {
namespace tile {
namespace hal {
namespace opencl {

// Implements hal::Result in terms of OpenCL events.
class Result final : public hal::Result {
 public:
  Result(const context::Context& ctx, std::shared_ptr<DeviceState> device_state, CLObj<cl_event> event);

  std::chrono::high_resolution_clock::duration GetDuration() const final;
  void LogStatistics() const final;

 private:
  context::Context ctx_;
  std::shared_ptr<DeviceState> device_state_;
  CLObj<cl_event> event_;
};

}  // namespace opencl
}  // namespace hal
}  // namespace tile
}  // namespace vertexai
