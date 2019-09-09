// Copyright 2017-2018 Intel Corporation.

#include "tile/hal/cm/executor.h"

#include <string>
#include <utility>

#include "base/util/compat.h"
#include "base/util/env.h"
#include "base/util/error.h"
#include "tile/hal/cm/buffer.h"
#include "tile/hal/cm/compute_kernel.h"
#include "tile/hal/cm/err.h"
#include "tile/hal/cm/event.h"
#include "tile/hal/cm/executable.h"
#include "tile/hal/cm/info.h"
#include "tile/hal/cm/kernel.h"
#include "tile/hal/cm/library.h"
#include "tile/hal/cm/runtime.h"
#include "tile/hal/cm/zero_kernel.h"
#include "tile/hal/util/selector.h"

namespace vertexai {
namespace tile {
namespace hal {
namespace cm {

Executor::Executor(std::shared_ptr<DeviceState> device_state)
    : device_state_{device_state}, info_{GetHardwareInfo(device_state->info())} {
  InitSharedMemory();
}

std::shared_ptr<hal::Event> Executor::Copy(const context::Context& ctx, const std::shared_ptr<hal::Buffer>& from,
                                           std::size_t from_offset, const std::shared_ptr<hal::Buffer>& to,
                                           std::size_t to_offset, std::size_t length,
                                           const std::vector<std::shared_ptr<hal::Event>>& dependencies) {
  auto from_buf = Buffer::Downcast(from);
  auto to_buf = Buffer::Downcast(to);

  if (from_buf->size() <= from_offset || from_buf->size() < length || from_buf->size() < from_offset + length ||
      to_buf->size() <= to_offset || to_buf->size() < length || to_buf->size() < to_offset + length) {
    throw error::InvalidArgument{"Invalid copy request: from=" + std::to_string(from_buf->size()) +
                                 " bytes, from_offset=" + std::to_string(from_offset) + ", to=" +
                                 std::to_string(to_buf->size()) + " bytes, to_offset=" + std::to_string(to_offset) +
                                 ", length=" + std::to_string(length)};
  }

  context::Activity activity{ctx, "tile::hal::cm::Copy"};

  auto from_base = from_buf->base();
  auto from_ptr = from_buf->mem();
  auto to_base = to_buf->base();
  auto to_ptr = to_buf->mem();

  const auto& queue = device_state_->cmqueue();
  auto mdeps = Event::Downcast(dependencies, queue);

  if (from_base && to_base) {
    // Memory-to-memory copy.
    if (!mdeps.size()) {
      memcpy(static_cast<char*>(to_base) + to_offset, static_cast<char*>(from_base) + from_offset, length);
      CmEvent* cm_event;
      return std::make_shared<Event>(activity.ctx(), device_state_, cm_event, queue);
    }

    CmEvent* event;

    Event::WaitFor(dependencies, device_state_)
        .then([event, to_base, to_offset, from_base, from_offset,
               length](boost::shared_future<std::vector<std::shared_ptr<hal::Result>>> result) {
          try {
            result.get();
            memcpy(static_cast<char*>(to_base) + to_offset, static_cast<char*>(from_base) + from_offset, length);
          } catch (...) {
          }
        });
    return std::make_shared<Event>(activity.ctx(), device_state_, std::move(event), queue);
  }

  if (from_base && to_ptr) {
    CmEvent* event;
    return std::make_shared<Event>(activity.ctx(), device_state_, std::move(event), queue);
  }

  if (from_ptr && to_base) {
    CmEvent* event;
    auto result = std::make_shared<Event>(activity.ctx(), device_state_, std::move(event), queue);
    return result;
  }

  if (from_ptr && to_ptr) {
    CmEvent* event;
    auto result = std::make_shared<Event>(activity.ctx(), device_state_, std::move(event), queue);
    return result;
  }
  throw error::Unimplemented("Unable to copy data between the provided buffers");
}

boost::future<std::unique_ptr<hal::Executable>> Executor::Prepare(hal::Library* library) {
  Library* exe = Library::Downcast(library, device_state_);

  std::vector<std::unique_ptr<Kernel>> kernels;
  kernels.reserve(exe->kernel_ids().size());

  for (std::size_t kidx = 0; kidx < exe->kernel_ids().size(); ++kidx) {
    const lang::KernelInfo& kinfo = exe->kernel_info()[kidx];
    auto kid = exe->kernel_ids()[kidx];

    if (kinfo.ktype == lang::KernelType::kZero) {
      kernels.emplace_back(std::make_unique<ZeroKernel>(device_state_, kinfo, kid));
      continue;
    }

    std::string kname = kinfo.kname;
    CmKernel* kernel = NULL;
    CmDevice* pCmDev = device_state_->cmdev();
    CmProgram* program = exe->getProgramMap().at(kname);
    auto cm = exe->get_emit_map().at(kname);

    cm_result_check(pCmDev->CreateKernel(program, kname.c_str(), kernel));
    if (!kernel) {
      throw std::runtime_error(std::string("Unable to initialize cm kernel: "));
    }

    kernels.emplace_back(
        std::make_unique<ComputeKernel>(device_state_, std::move(kernel), exe->kernel_info()[kidx], kid, cm));
  }

  return boost::make_ready_future(std::unique_ptr<hal::Executable>(std::make_unique<Executable>(std::move(kernels))));
}

void Executor::Flush() { device_state_->FlushCommandQueue(); }

boost::future<std::vector<std::shared_ptr<hal::Result>>> Executor::WaitFor(
    const std::vector<std::shared_ptr<hal::Event>>& events) {
  return Event::WaitFor(events, device_state_);
}

}  // namespace cm
}  // namespace hal
}  // namespace tile
}  // namespace vertexai
