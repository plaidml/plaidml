// Copyright 2017, Vertex.AI.

#pragma once

#include <memory>

#include "tile/base/hal.h"
#include "tile/hal/opencl/device_state.h"
#include "tile/hal/opencl/ocl.h"

namespace vertexai {
namespace tile {
namespace hal {
namespace opencl {

struct ResultInfo {
  cl_ulong queued_time = 0;
  cl_ulong submit_time = 0;
  cl_ulong start_time = 0;
  cl_ulong end_time = 0;
  cl_int status = 0;
  std::chrono::high_resolution_clock::duration execution_duration{std::chrono::high_resolution_clock::duration::zero()};
};

// Implements hal::Result in terms of OpenCL events.
class Result final : public hal::Result {
 public:
  Result(const context::Context& ctx, std::shared_ptr<DeviceState> device_state, CLObj<cl_event> event,
         const DeviceState::Queue& queue);

  std::chrono::high_resolution_clock::duration GetDuration() const final;
  void LogStatistics() const final;

 private:
  context::Context ctx_;
  std::shared_ptr<DeviceState> device_state_;
  CLObj<cl_event> event_;
  bool profiling_enabled_;
  mutable std::unique_ptr<ResultInfo> info_;
  mutable std::once_flag once_;
};

class KernelResult final : public hal::Result {
 public:
  KernelResult(const context::Context& ctx, std::shared_ptr<DeviceState> device_state, CLObj<cl_event> event,
               const lang::KernelInfo& ki, const DeviceState::Queue& queue);

  std::chrono::high_resolution_clock::duration GetDuration() const final;
  void LogStatistics() const final;

 private:
  context::Context ctx_;
  std::shared_ptr<DeviceState> device_state_;
  CLObj<cl_event> event_;
  bool profiling_enabled_;
  mutable std::unique_ptr<ResultInfo> info_;
  mutable std::once_flag once_;

  lang::KernelInfo ki_;
};

}  // namespace opencl
}  // namespace hal
}  // namespace tile
}  // namespace vertexai
