// Copyright 2017, Vertex.AI. CONFIDENTIAL

#include "tile/hal/opencl/executor.h"

#include <string>
#include <utility>

#include "base/util/compat.h"
#include "base/util/error.h"
#include "tile/hal/opencl/buffer.h"
#include "tile/hal/opencl/event.h"
#include "tile/hal/opencl/global_memory.h"
#include "tile/hal/opencl/info.h"
#include "tile/hal/opencl/kernel.h"
#include "tile/hal/opencl/library.h"
#include "tile/hal/opencl/local_memory.h"
#include "tile/hal/util/selector.h"

namespace vertexai {
namespace tile {
namespace hal {
namespace opencl {

Executor::Executor(const std::shared_ptr<DeviceState>& device_state)
    : device_state_{device_state}, info_{GetHardwareInfo(device_state->info())} {
  InitSharedMemory();

  if (device_state_->info().local_mem_type() == proto::LocalMemType::Local) {
    VLOG(3) << "Enabling OpenCL device-local memory";
    device_memory_ = compat::make_unique<LocalMemory>(device_state_);
  }
}

std::shared_ptr<hal::Event> Executor::Copy(const context::Context& ctx, const std::shared_ptr<hal::Buffer>& from,
                                           std::size_t from_offset, const std::shared_ptr<hal::Buffer>& to,
                                           std::size_t to_offset, std::size_t length,
                                           const std::vector<std::shared_ptr<hal::Event>>& dependencies) {
  auto from_buf = Buffer::Upcast(from, device_state_->cl_ctx());
  auto to_buf = Buffer::Upcast(to, device_state_->cl_ctx());

  if (from_buf->size() <= from_offset || from_buf->size() < length || from_buf->size() < from_offset + length ||
      to_buf->size() <= to_offset || to_buf->size() < length || to_buf->size() < to_offset + length) {
    throw error::InvalidArgument{"Invalid copy request"};
  }

  context::Activity activity{ctx, "tile::hal::opencl::Copy"};

  auto from_base = from_buf->base();
  auto from_ptr = from_buf->mem();
  auto to_base = to_buf->base();
  auto to_ptr = to_buf->mem();

  const auto& queue = device_state_->cl_queue(activity.ctx().is_logging_events());
  auto mdeps = Event::Upcast(dependencies, device_state_->cl_ctx(), queue);

  if (from_base && to_base) {
    // Memory-to-memory copy.
    if (!mdeps.size()) {
      memcpy(static_cast<char*>(to_base) + to_offset, static_cast<char*>(from_base) + from_offset, length);
      return std::make_shared<Event>(activity.ctx(), device_state_, CLObj<cl_event>(), queue);
    }

    Err err;
    CLObj<cl_event> event = clCreateUserEvent(device_state_->cl_ctx().get(), err.ptr());
    Err::Check(err, "Unable to allocate a synchronization event");

    Event::WaitFor(dependencies, device_state_)
        .then([event, to_base, to_offset, from_base, from_offset, length](boost::future<void> result) {
          try {
            result.get();
            memcpy(static_cast<char*>(to_base) + to_offset, static_cast<char*>(from_base) + from_offset, length);
            clSetUserEventStatus(event.get(), CL_SUCCESS);
          } catch (...) {
            clSetUserEventStatus(event.get(), CL_OUT_OF_RESOURCES);
          }
        });

    return std::make_shared<Event>(activity.ctx(), device_state_, std::move(event), queue);
  }

  if (from_base && to_ptr) {
    // Memory-to-buffer write
    CLObj<cl_event> event;
    Err err = clEnqueueWriteBuffer(queue.cl_queue.get(),                         // command_queue
                                   to_ptr,                                       // buffer
                                   CL_FALSE,                                     // blocking_write
                                   to_offset,                                    // offset
                                   length,                                       // size
                                   static_cast<char*>(from_base) + from_offset,  // ptr
                                   mdeps.size(),                                 // num_events_in_wait_list
                                   mdeps.size() ? mdeps.data() : nullptr,        // event_wait_list
                                   event.LvaluePtr());                           // event
    Err::Check(err, "Unable to write to the destination buffer");
    auto result = std::make_shared<Event>(activity.ctx(), device_state_, std::move(event), queue);
    queue.Flush();
    return result;
  }

  if (from_ptr && to_base) {
    // Buffer-to-memory read
    CLObj<cl_event> event;
    Err err = clEnqueueReadBuffer(queue.cl_queue.get(),                     // command_queue
                                  from_ptr,                                 // buffer
                                  CL_FALSE,                                 // blocking_read
                                  from_offset,                              // offset
                                  length,                                   // size
                                  static_cast<char*>(to_base) + to_offset,  // ptr
                                  mdeps.size(),                             // num_events_in_wait_list
                                  mdeps.size() ? mdeps.data() : nullptr,    // event_wait_list
                                  event.LvaluePtr());                       // event
    Err::Check(err, "Unable to read from the source buffer");
    auto result = std::make_shared<Event>(activity.ctx(), device_state_, std::move(event), queue);
    queue.Flush();
    return result;
  }

  if (from_ptr && to_ptr) {
    // Buffer-to-buffer copy
    CLObj<cl_event> event;
    Err err = clEnqueueCopyBuffer(queue.cl_queue.get(),                   // command_queue
                                  from_ptr,                               // src_buffer
                                  to_ptr,                                 // dst_buffer
                                  from_offset,                            // src_offset
                                  to_offset,                              // dst_offset
                                  length,                                 // size
                                  mdeps.size(),                           // num_events_in_wait_list
                                  mdeps.size() ? mdeps.data() : nullptr,  // event_wait_list
                                  event.LvaluePtr());                     // event
    Err::Check(err, "Unable to copy data between the provided buffers");
    auto result = std::make_shared<Event>(activity.ctx(), device_state_, std::move(event), queue);
    queue.Flush();
    return result;
  }

  // This should never happen.
  // If it does, perhaps we could map both buffers (discarding the target's existing data) and
  // memcpy.
  throw error::Unimplemented("Unable to copy data between the provided buffers");
}

boost::future<std::unique_ptr<hal::Kernel>> Executor::Prepare(hal::Library* library, std::size_t kernel_index) {
  Library* exe = Library::Upcast(library, device_state_);

  Err err;
  std::string kname = exe->kernel_info()[kernel_index].kname;
  CLObj<cl_kernel> kernel = clCreateKernel(exe->program().get(), kname.c_str(), err.ptr());
  if (!kernel) {
    throw std::runtime_error(std::string("Unable to initialize OpenCL kernel: ") + err.str());
  }

  boost::uuids::uuid kernel_uuid = exe->kernel_uuids()[kernel_index];
  return boost::make_ready_future(std::unique_ptr<hal::Kernel>(
      compat::make_unique<Kernel>(device_state_, std::move(kernel), exe->kernel_info()[kernel_index], kernel_uuid)));
}

void Executor::Flush() { device_state_->FlushCommandQueue(); }

}  // namespace opencl
}  // namespace hal
}  // namespace tile
}  // namespace vertexai
