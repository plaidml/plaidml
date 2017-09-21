// Copyright 2017, Vertex.AI. CONFIDENTIAL

#include "tile/hal/opencl/device_state.h"

#include <string>
#include <utility>
#include <vector>

#include "base/util/error.h"
#include "tile/hal/opencl/info.h"
#include "tile/hal/util/selector.h"

namespace vertexai {
namespace tile {
namespace hal {
namespace opencl {
namespace {

DeviceState::Queue MakeQueue(cl_device_id did, const CLObj<cl_context>& cl_ctx,
                             cl_command_queue_properties extra_properties = 0) {
  Err err;

  // Probe the device for supported queue properties.
  auto props = CLInfoType<CL_DEVICE_QUEUE_PROPERTIES>::Read(did);

  // Enable out of order execution if supported.
  // Clear any properties we don't understand (aka everything else).
  props &= CL_QUEUE_OUT_OF_ORDER_EXEC_MODE_ENABLE;

  DeviceState::Queue result;
  result.props = props | extra_properties;

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
    : did_{did},
      config_{config},
      info_{std::move(info)},
      cl_ctx_{cl_ctx},
      cl_normal_queue_(MakeQueue(did_, cl_ctx_)),
      cl_profiling_queue_(MakeQueue(did_, cl_ctx_, CL_QUEUE_PROFILING_ENABLE)),
      uuid_(ctx.activity_uuid()) {}

void DeviceState::FlushCommandQueue() {
  cl_normal_queue_.Flush();
  cl_profiling_queue_.Flush();
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
