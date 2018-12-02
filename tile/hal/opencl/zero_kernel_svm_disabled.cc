// Copyright 2017-2018 Intel Corporation.

#include "tile/hal/opencl/zero_kernel.h"

namespace vertexai {
namespace tile {
namespace hal {
namespace opencl {

CLObj<cl_event> ZeroKernel::FillBufferImpl(const DeviceState::Queue& queue, Buffer* buf, void* pattern,
                                           size_t pattern_size, const std::vector<cl_event>& deps) {
  CLObj<cl_event> done;
  auto event_wait_list = deps.size() ? deps.data() : nullptr;
  Err err = clEnqueueFillBuffer(queue.cl_queue.get(), buf->mem(), pattern, pattern_size, 0, buf->size(), deps.size(),
                                event_wait_list, done.LvaluePtr());
  Err::Check(err, "unable to fill buffer");
  return done;
}

}  // namespace opencl
}  // namespace hal
}  // namespace tile
}  // namespace vertexai
