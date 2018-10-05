// Copyright 2017-2018 Intel Corporation.

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
  cl_ulong queued_time;
  cl_ulong submit_time;
  cl_ulong start_time;
  cl_ulong end_time;
  cl_int status;
  std::chrono::high_resolution_clock::duration execution_duration;
};

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
  mutable std::unique_ptr<ResultInfo> info_;
  mutable std::once_flag once_;
};

class KernelResult final : public hal::Result {
 public:
  KernelResult(const context::Context& ctx, std::shared_ptr<DeviceState> device_state, CLObj<cl_event> event,
               const lang::KernelInfo& ki);

  std::chrono::high_resolution_clock::duration GetDuration() const final;
  void LogStatistics() const final;

 private:
  context::Context ctx_;
  std::shared_ptr<DeviceState> device_state_;
  CLObj<cl_event> event_;
  mutable std::unique_ptr<ResultInfo> info_;
  mutable std::once_flag once_;

  lang::KernelInfo ki_;
};

}  // namespace opencl
}  // namespace hal
}  // namespace tile
}  // namespace vertexai
