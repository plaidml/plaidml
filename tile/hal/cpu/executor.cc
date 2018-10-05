// Copyright 2017-2018 Intel Corporation.

#include "tile/hal/cpu/executor.h"

#include <string>
#include <utility>

#include "base/util/compat.h"
#include "base/util/error.h"
#include "tile/hal/cpu/buffer.h"
#include "tile/hal/cpu/event.h"
#include "tile/hal/cpu/executable.h"
#include "tile/hal/cpu/library.h"
#include "tile/hal/cpu/memory.h"
#include "tile/hal/util/selector.h"

namespace vertexai {
namespace tile {
namespace hal {
namespace cpu {
namespace {

hal::proto::HardwareInfo GetHardwareInfo() {
  // Get the info required to tell the compiler how to generate efficient code for the target hardware.
  hal::proto::HardwareInfo info;

  // TODO: We should use the actual processor identifier here.
  info.set_type(hal::proto::HardwareType::CPU);
  info.set_name("LLVM CPU");
  info.set_vendor("LLVM");

  hal::proto::HardwareSettings* settings = info.mutable_settings();

  // We will run one thread per CPU core to process one workgroup. That means we will have a single thread per
  // workgroup.
  settings->set_threads(1);

  // The vector size is currently the number of elements in the a SIMD register for some assumed datatype. We should
  // probably change this to indicate the bit width of the register instead, because the number of elements depends on
  // element type.
  settings->set_vec_size(4);

  // GPUs have a concept of local memory, which works like an L1 cache that you manage explicitly. We'll let the
  // processor manage cache for us, which means we are using "global memory" in GPU terms.
  settings->set_use_global(true);

  // Memory width is the size of a cache line. That is, what is the smallest unit of memory we can load at a time? This
  // is technically variable but it happens to be the same for x86 and ARM architectures we care about.

  settings->set_mem_width(64);

  // Maximum memory is another concept based on GPU local memory. It roughly means the size of the L1 cache: that is,
  // how much data can we efficiently read at one time?

  settings->set_max_mem(32768);

  // Maximum number of registers refers to the vector unit registers. It controls the number of outputs which can be
  // generated at a time. This number is for NEON. 32-bit SSE had 8; AMD64 extended it to 16.
  settings->set_max_regs(32);

  // Minimum number of work groups: we need one workgroup per core.
  settings->set_goal_groups(1);

  // No idea what flops per byte means.
  // TODO: Fill this in with a more correct value.
  settings->set_goal_flops_per_byte(1);

  // goal dimension sizes... still no idea what this does
  // TODO: Fill this in with a more correct value.
  settings->add_dim_sizes(0);

  return info;
}

}  // namespace

Executor::Executor() : info_{GetHardwareInfo()}, memory_{new Memory()} {}

std::shared_ptr<hal::Event> Executor::Copy(const context::Context& ctx, const std::shared_ptr<hal::Buffer>& from,
                                           std::size_t from_offset, const std::shared_ptr<hal::Buffer>& to,
                                           std::size_t to_offset, std::size_t length,
                                           const std::vector<std::shared_ptr<hal::Event>>& dependencies) {
  auto f = Buffer::Downcast(from);
  auto t = Buffer::Downcast(to);
  if (f->size() <= from_offset || f->size() < length || f->size() < from_offset + length || t->size() <= to_offset ||
      t->size() < length || t->size() < to_offset + length) {
    throw error::InvalidArgument{"Invalid copy request"};
  }
  auto deps = Event::WaitFor(dependencies);
  context::Context ctx_copy{ctx};
  auto evt = deps.then([ctx = std::move(ctx_copy), f, t, from_offset, to_offset,
                        length](decltype(deps) fut) -> std::shared_ptr<hal::Result> {
    fut.get();
    char* fb = static_cast<char*>(f->base()) + to_offset;
    char* tb = static_cast<char*>(t->base()) + from_offset;
    auto start = std::chrono::high_resolution_clock::now();
    memcpy(tb, fb, length);
    return std::make_shared<Result>(ctx, "tile::hal::cpu::CopyMemory", start,
                                    std::chrono::high_resolution_clock::now());
  });
  return std::make_shared<cpu::Event>(std::move(evt));
}

boost::future<std::unique_ptr<hal::Executable>> Executor::Prepare(hal::Library* library) {
  auto lib = Library::Downcast(library);
  auto k = compat::make_unique<cpu::Executable>(lib->engines(), lib->kernels());
  return boost::make_ready_future(std::unique_ptr<hal::Executable>(std::move(k)));
}

boost::future<std::vector<std::shared_ptr<hal::Result>>> Executor::WaitFor(
    const std::vector<std::shared_ptr<hal::Event>>& events) {
  return Event::WaitFor(events);
}

}  // namespace cpu
}  // namespace hal
}  // namespace tile
}  // namespace vertexai
