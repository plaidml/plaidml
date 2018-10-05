// Copyright 2017-2018 Intel Corporation.

#pragma once

#include <memory>

#include "base/context/context.h"
#include "tile/hal/opencl/ocl.h"
#include "tile/hal/opencl/opencl.pb.h"
#include "tile/lang/generate.h"

namespace vertexai {
namespace tile {
namespace hal {
namespace opencl {

// DeviceState represents the state of a device, including all OpenCL objects needed to control the device.
class DeviceState {
 public:
  struct Queue {
    CLObj<cl_command_queue> cl_queue;
    cl_command_queue_properties props;

    void Flush() const;
  };

  DeviceState(const context::Context& ctx, const CLObj<cl_context>& cl_ctx, cl_device_id did, proto::DeviceInfo info);

  void Initialize(const hal::proto::HardwareSettings& settings);

  const cl_device_id did() const { return did_; }
  const CLObj<cl_context>& cl_ctx() const { return cl_ctx_; }
  const proto::DeviceInfo& info() const { return info_; }
  const Queue& cl_normal_queue() const { return *cl_normal_queue_; }
  const Queue& cl_profiling_queue() const { return *cl_profiling_queue_; }
  const Queue& cl_queue(bool enable_profiling) const {
    if (enable_profiling) {
      return cl_profiling_queue();
    }
    return cl_normal_queue();
  }
  const context::Clock& clock() const { return clock_; }
  const context::proto::ActivityID& id() const { return id_; }

  cl_map_flags map_discard_flags() const;

  void FlushCommandQueue();

  bool HasDeviceExtension(const char* extension);

 private:
  const cl_device_id did_;
  const proto::DeviceInfo info_;
  const CLObj<cl_context> cl_ctx_;
  std::unique_ptr<const Queue> cl_normal_queue_;
  std::unique_ptr<const Queue> cl_profiling_queue_;
  const context::Clock clock_;
  const context::proto::ActivityID id_;
};

}  // namespace opencl
}  // namespace hal
}  // namespace tile
}  // namespace vertexai
