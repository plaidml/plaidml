// Copyright 2017-2018 Intel Corporation.

#include "tile/hal/cm/zero_kernel.h"

#include "base/util/error.h"
#include "tile/hal/cm/event.h"

namespace vertexai {
namespace tile {
namespace hal {
namespace cm {

ZeroKernel::ZeroKernel(const std::shared_ptr<DeviceState>& device_state, const lang::KernelInfo& kinfo,
                       context::proto::ActivityID kid)
    : device_state_{device_state}, kinfo_{kinfo}, kid_(kid) {}

std::shared_ptr<hal::Event> ZeroKernel::Run(const context::Context& ctx,
                                            const std::vector<std::shared_ptr<hal::Buffer>>& params,
                                            const std::vector<std::shared_ptr<hal::Event>>& dependencies,
                                            bool enable_profiling) {
  const auto& queue = device_state_->cmqueue();
  auto deps = Event::Downcast(dependencies, queue);
  IVLOG(4, "Running zero-fill memory " << kinfo_.kname);

  if (params.size() != 1) {
    throw error::Internal("Zero-memory operation invoked with a memory region count != 1");
  }

  Buffer* buf = Buffer::Downcast(params[0].get());
  IVLOG(4, "  Buffer: " << buf);

  context::Activity activity{ctx, "tile::hal::cm::Buffer::Fill"};
  proto::RunInfo rinfo;
  *rinfo.mutable_kernel_id() = kid_;
  activity.AddMetadata(rinfo);

  CmEvent* done;
  return std::make_shared<Event>(activity.ctx(), device_state_, done, queue);
}

}  // namespace cm
}  // namespace hal
}  // namespace tile
}  // namespace vertexai
