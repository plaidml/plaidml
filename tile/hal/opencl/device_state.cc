// Copyright 2017, Vertex.AI.

#include "tile/hal/opencl/device_state.h"

#include <string>
#include <utility>
#include <vector>

#include "base/util/env.h"
#include "base/util/error.h"
#include "tile/hal/opencl/info.h"
#include "tile/hal/util/selector.h"

namespace vertexai {
namespace tile {
namespace hal {
namespace opencl {
namespace {

DeviceState::Queue MakeQueue(const context::Context& ctx, cl_device_id did, const CLObj<cl_context>& cl_ctx,
                             const proto::Driver& config, const hal::proto::HardwareSettings& settings) {
  Err err;

  cl_command_queue_properties mask = 0;
  if (!settings.is_synchronous()) {
    // Enable out of order execution if supported.
    mask = CL_QUEUE_OUT_OF_ORDER_EXEC_MODE_ENABLE;
  }

  // Probe the device for supported queue properties.
  // Clear any properties we don't understand (aka everything else).
  DeviceState::Queue result;
  result.props = CLInfoType<CL_DEVICE_QUEUE_PROPERTIES>::Read(did) & mask;
  if (config.enable_profiling() || env::GetVar("PLAIDML_PROFILING_ENABLED").length() || ctx.is_logging_events() ||
      VLOG_IS_ON(1)) {
    result.props |= CL_QUEUE_PROFILING_ENABLE;
  }

  result.cl_queue = clCreateCommandQueue(cl_ctx.get(), did, result.props, err.ptr());
  if (!result.cl_queue) {
    throw std::runtime_error(std::string("creating a command queue for an OpenCL device: ") + err.str());
  }
  return result;
}

}  // namespace

void DeviceState::Queue::Flush() const { Err::Check(clFlush(cl_queue.get()), "Unable to flush command queue"); }

DeviceState::DeviceState(const context::Context& ctx, const CLObj<cl_context>& cl_ctx, cl_device_id did,
                         const std::shared_ptr<proto::Driver>& config, proto::DeviceInfo info)
    : did_{did}, config_{config}, info_{std::move(info)}, cl_ctx_{cl_ctx}, uuid_(ctx.activity_uuid()) {}

void DeviceState::Initialize(const context::Context& ctx, const hal::proto::HardwareSettings& settings) {
  cl_queue_ = compat::make_unique<Queue>(MakeQueue(ctx, did_, cl_ctx_, *config_, settings));
}

void DeviceState::FlushCommandQueue() { cl_queue_->Flush(); }

bool DeviceState::HasDeviceExtension(const char* extension) {
  for (auto ext : info_.extension()) {
    if (ext == extension) {
      return true;
    }
  }
  return false;
}

cl_map_flags DeviceState::map_discard_flags() const {
  // TODO: parse this string out into different parts and then check major/minor directly
  if (info_.version() == "OpenCL 1.1 ") {
    return CL_MAP_WRITE;
  }
  return CL_MAP_WRITE_INVALIDATE_REGION;
}

}  // namespace opencl
}  // namespace hal
}  // namespace tile
}  // namespace vertexai
