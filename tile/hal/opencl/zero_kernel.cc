// Copyright 2017-2018 Intel Corporation.

#include "tile/hal/opencl/zero_kernel.h"

#include "base/util/error.h"
#include "tile/hal/opencl/event.h"

namespace vertexai {
namespace tile {
namespace hal {
namespace opencl {

ZeroKernel::ZeroKernel(const std::shared_ptr<DeviceState>& device_state, const lang::KernelInfo& kinfo,
                       context::proto::ActivityID kid)
    : device_state_{device_state}, kinfo_{kinfo}, kid_(kid) {}

std::shared_ptr<hal::Event> ZeroKernel::Run(const context::Context& ctx,
                                            const std::vector<std::shared_ptr<hal::Buffer>>& params,
                                            const std::vector<std::shared_ptr<hal::Event>>& dependencies,
                                            bool enable_profiling) {
  const auto& queue = device_state_->cl_queue(enable_profiling);
  auto deps = Event::Downcast(dependencies, device_state_->cl_ctx(), queue);
  IVLOG(4, "Running zero-fill memory " << kinfo_.kname);

  if (params.size() != 1) {
    throw error::Internal("Zero-memory operation invoked with a memory region count != 1");
  }

  Buffer* buf = Buffer::Downcast(params[0].get(), device_state_->cl_ctx());
  IVLOG(4, "  Buffer: " << buf);

  if (VLOG_IS_ON(4)) {
    VLOG(4) << "  Deps.size(): " << deps.size();
    for (auto dep : deps) {
      VLOG(4) << "  Dep: " << dep;
    }
  }

  context::Activity activity{ctx, "tile::hal::opencl::Buffer::Fill"};
  proto::RunInfo rinfo;
  *rinfo.mutable_kernel_id() = kid_;
  activity.AddMetadata(rinfo);

  cl_uchar char_pattern = 0;
  cl_ulong long_pattern = 0;
  void* pattern = &long_pattern;
  size_t pattern_size = sizeof(long_pattern);
  if (buf->size() % sizeof(long_pattern) != 0) {
    pattern = &char_pattern;
    pattern_size = sizeof(char_pattern);
  }

  CLObj<cl_event> done = FillBufferImpl(queue, buf, pattern, pattern_size, deps);

  IVLOG(4, "  Produced dep: " << done.get());

  return std::make_shared<Event>(activity.ctx(), device_state_, done, queue);
}

}  // namespace opencl
}  // namespace hal
}  // namespace tile
}  // namespace vertexai
