// Copyright 2017-2018 Intel Corporation.

#include "tile/hal/opencl/cl_mem_buffer.h"

#include <utility>

#include "tile/hal/opencl/event.h"
#include "tile/hal/opencl/result.h"

namespace vertexai {
namespace tile {
namespace hal {
namespace opencl {

CLMemBuffer::CLMemBuffer(const std::shared_ptr<DeviceState>& device_state, std::uint64_t size, CLObj<cl_mem> mem)
    : Buffer{device_state->cl_ctx(), size}, device_state_{device_state}, mem_{std::move(mem)} {}

void CLMemBuffer::SetKernelArg(const CLObj<cl_kernel>& kernel, std::size_t index) {
  cl_mem m = mem_.get();
  Err err = clSetKernelArg(kernel.get(),    // kernel
                           index,           // arg_index
                           sizeof(cl_mem),  // arg_size
                           &m);             // arg_value
  Err::Check(err, "Unable to set a kernel memory pointer");
}

boost::future<void*> CLMemBuffer::MapCurrent(const std::vector<std::shared_ptr<hal::Event>>& deps) {
  const auto& queue = device_state_->cl_normal_queue();
  auto mdeps = Event::Downcast(deps, device_state_->cl_ctx(), queue);
  Err err;
  // TODO: Create a way for MapCurrent to specify read, write, or both, so that when we're reading (the usual case for
  // MapCurrent) we don't write the buffer back to the device when we're done (although hopefully the implementation
  // uses the dirty bits to elide that case for us).
  base_ = clEnqueueMapBuffer(queue.cl_queue.get(),                   // command_queue
                             mem_.get(),                             // buffer
                             CL_TRUE,                                // blocking_map
                             CL_MAP_READ | CL_MAP_WRITE,             // map_flags
                             0,                                      // offset
                             size(),                                 // size
                             mdeps.size(),                           // num_events_in_wait_list
                             mdeps.size() ? mdeps.data() : nullptr,  // event_wait_list
                             nullptr,                                // event
                             err.ptr());                             // errcode_ret
  Err::Check(err, "Unable to map memory");
  return boost::make_ready_future(base_);
}

boost::future<void*> CLMemBuffer::MapDiscard(const std::vector<std::shared_ptr<hal::Event>>& deps) {
  const auto& queue = device_state_->cl_normal_queue();
  auto mdeps = Event::Downcast(deps, device_state_->cl_ctx(), queue);
  Err err;
  base_ = clEnqueueMapBuffer(queue.cl_queue.get(),                   // command_queue
                             mem_.get(),                             // buffer
                             CL_TRUE,                                // blocking_map
                             device_state_->map_discard_flags(),     // map_flags
                             0,                                      // offset
                             size(),                                 // size
                             mdeps.size(),                           // num_events_in_wait_list
                             mdeps.size() ? mdeps.data() : nullptr,  // event_wait_list
                             nullptr,                                // event
                             err.ptr());                             // errcode_ret
  Err::Check(err, "Unable to map memory");
  return boost::make_ready_future(base_);
}

std::shared_ptr<hal::Event> CLMemBuffer::Unmap(const context::Context& ctx) {
  const auto& queue = device_state_->cl_normal_queue();
  context::Activity activity{ctx, "tile::hal::opencl::Buffer::Unmap"};
  CLObj<cl_event> evt;
  Err err = clEnqueueUnmapMemObject(queue.cl_queue.get(),  // command_queue
                                    mem_.get(),            // memobj
                                    base_,                 // mapped_ptr
                                    0,                     // num_events_in_wait_list
                                    nullptr,               // event_wait_list
                                    evt.LvaluePtr());      // event
  Err::Check(err, "Unable to unmap memory");
  base_ = nullptr;
  auto result = std::make_shared<Event>(activity.ctx(), device_state_, std::move(evt), queue);
  queue.Flush();
  return result;
}

}  // namespace opencl
}  // namespace hal
}  // namespace tile
}  // namespace vertexai
