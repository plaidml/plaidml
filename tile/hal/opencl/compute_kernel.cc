// Copyright 2017-2018 Intel Corporation.

#include "tile/hal/opencl/compute_kernel.h"

#include <utility>

#include "base/util/error.h"
#include "base/util/uuid.h"
#include "tile/hal/opencl/buffer.h"
#include "tile/hal/opencl/event.h"
#include "tile/hal/opencl/result.h"

namespace vertexai {
namespace tile {
namespace hal {
namespace opencl {

ComputeKernel::ComputeKernel(const std::shared_ptr<DeviceState>& device_state, CLObj<cl_kernel> kernel,
                             const lang::KernelInfo& info, context::proto::ActivityID kernel_id)
    : device_state_{device_state}, kernel_{std::move(kernel)}, ki_(info), kernel_id_(kernel_id) {
  if (VLOG_IS_ON(3)) {
    size_t work_group_size;
    Err::Check(clGetKernelWorkGroupInfo(kernel_.get(), device_state_->did(), CL_KERNEL_WORK_GROUP_SIZE,
                                        sizeof(work_group_size), &work_group_size, nullptr),
               "reading kernel work group size");
    VLOG(5) << "Kernel \"" << ki_.kname << "\": WorkGroupSize:  " << work_group_size;

    size_t sizes[3];
    Err::Check(clGetKernelWorkGroupInfo(kernel_.get(), device_state_->did(), CL_KERNEL_COMPILE_WORK_GROUP_SIZE,
                                        sizeof(sizes), &sizes, nullptr),
               "reading kernel compile work group size");
    VLOG(5) << "Kernel \"" << ki_.kname << "\": CompWorkSize:   [" << sizes[0] << ", " << sizes[1] << ", " << sizes[2]
            << "]";

    cl_ulong local_mem_size;
    Err::Check(clGetKernelWorkGroupInfo(kernel_.get(), device_state_->did(), CL_KERNEL_LOCAL_MEM_SIZE,
                                        sizeof(local_mem_size), &local_mem_size, nullptr),
               "reading kernel local memory size");
    VLOG(5) << "Kernel \"" << ki_.kname << "\": LocalMemSize:   " << local_mem_size;

    size_t pref_work_group_size;
    Err::Check(
        clGetKernelWorkGroupInfo(kernel_.get(), device_state_->did(), CL_KERNEL_PREFERRED_WORK_GROUP_SIZE_MULTIPLE,
                                 sizeof(pref_work_group_size), &pref_work_group_size, nullptr),
        "reading kernel preferred work group size multiple");
    VLOG(5) << "Kernel \"" << ki_.kname << "\": PrefWorkGpMult: " << pref_work_group_size;

    cl_ulong priv_mem_size;
    Err::Check(clGetKernelWorkGroupInfo(kernel_.get(), device_state_->did(), CL_KERNEL_PRIVATE_MEM_SIZE,
                                        sizeof(priv_mem_size), &priv_mem_size, nullptr),
               "reading kernel private memory size");
    VLOG(5) << "Kernel \"" << ki_.kname << "\": PrivateMemSize: " << priv_mem_size;
  }
}

std::shared_ptr<hal::Event> ComputeKernel::Run(const context::Context& ctx,
                                               const std::vector<std::shared_ptr<hal::Buffer>>& params,
                                               const std::vector<std::shared_ptr<hal::Event>>& dependencies,
                                               bool enable_profiling) {
  const auto& queue = device_state_->cl_queue(enable_profiling);
  auto deps = Event::Downcast(dependencies, device_state_->cl_ctx(), queue);
  VLOG(4) << "Running kernel " << ki_.kname;

  std::lock_guard<std::mutex> lock{mu_};
  for (std::size_t i = 0; i < params.size(); ++i) {
    Buffer* buf = Buffer::Downcast(params[i].get(), device_state_->cl_ctx());
    VLOG(4) << "  Param: " << buf;
    buf->SetKernelArg(kernel_, i);
  }

  if (VLOG_IS_ON(4)) {
    VLOG(4) << "  Deps.size(): " << deps.size();
    for (auto dep : deps) {
      VLOG(4) << "  Dep: " << dep;
    }
  }

  context::Activity activity{ctx, "tile::hal::opencl::Kernel::Run"};
  if (ctx.is_logging_events()) {
    proto::RunInfo rinfo;
    *rinfo.mutable_kernel_id() = kernel_id_;
    activity.AddMetadata(rinfo);
  }
  CLObj<cl_event> done;
  auto local_work_size = ki_.lwork[0] ? ki_.lwork.data() : nullptr;
  auto event_wait_list = deps.size() ? deps.data() : nullptr;
  Err err = clEnqueueNDRangeKernel(queue.cl_queue.get(),  // command_queue
                                   kernel_.get(),         // kernel
                                   3,                     // work_dim
                                   nullptr,               // global_work_offset
                                   ki_.gwork.data(),      // global_work_size
                                   local_work_size,       // local_work_size
                                   deps.size(),           // num_events_in_wait_list
                                   event_wait_list,       // event_wait_list
                                   done.LvaluePtr());     // event
  Err::Check(err, "unable to run OpenCL kernel");

  VLOG(4) << "  Produced dep: " << done.get();

  auto result = std::make_shared<KernelResult>(activity.ctx(), device_state_, done, ki_);
  return std::make_shared<Event>(activity.ctx(), device_state_, std::move(done), queue, std::move(result));
}

}  // namespace opencl
}  // namespace hal
}  // namespace tile
}  // namespace vertexai
