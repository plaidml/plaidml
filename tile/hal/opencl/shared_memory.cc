// Copyright 2017-2018 Intel Corporation.

#include <cstdint>
#include <mutex>
#include <vector>

#include "base/util/compat.h"
#include "base/util/error.h"
#include "base/util/logging.h"
#include "tile/base/hal.h"
#include "tile/hal/opencl/buffer.h"
#include "tile/hal/opencl/device_state.h"
#include "tile/hal/opencl/event.h"
#include "tile/hal/opencl/executor.h"
#include "tile/hal/opencl/ocl.h"

namespace vertexai {
namespace tile {
namespace hal {
namespace opencl {
namespace {

// Class declarations

class SharedArena final : public Arena, public std::enable_shared_from_this<SharedArena> {
 public:
  SharedArena(const std::shared_ptr<DeviceState>& device_state, std::uint64_t size);
  virtual ~SharedArena();

  std::shared_ptr<hal::Buffer> MakeBuffer(std::uint64_t offset, std::uint64_t size) final;

  const std::shared_ptr<DeviceState> device_state() const { return device_state_; }

 private:
  // A lock to guard clSVMAlloc/clSVMFree calls.  This shouldn't be necessary, but
  // it turns out we see crashes without it.
  static std::mutex svm_mu;

  const std::shared_ptr<DeviceState> device_state_;
  void* base_ = nullptr;
  std::uint64_t size_;
};

class SharedBuffer final : public Buffer {
 public:
  SharedBuffer(std::shared_ptr<SharedArena> arena, void* base, std::uint64_t size);

  void SetKernelArg(const CLObj<cl_kernel>& kernel, std::size_t index) final;

  boost::future<void*> MapCurrent(const std::vector<std::shared_ptr<hal::Event>>& deps) final;
  boost::future<void*> MapDiscard(const std::vector<std::shared_ptr<hal::Event>>& deps) final;
  std::shared_ptr<hal::Event> Unmap(const context::Context& ctx) final;

  void* base() const final { return base_; }

 private:
  std::shared_ptr<SharedArena> arena_;
  void* base_ = nullptr;
  std::uint64_t size_;
};

class SharedMemory final : public Memory {
 public:
  explicit SharedMemory(const std::shared_ptr<DeviceState>& device_state);

  std::uint64_t size_goal() const final {
    // TODO: Actually query the system physical memory size.
    return 16 * std::giga::num;
  }

  BufferAccessMask AllowedAccesses() const final { return BufferAccessMask::ALL; }

  std::size_t ArenaBufferAlignment() const final { return device_state_->info().mem_base_addr_align(); }

  std::shared_ptr<hal::Buffer> MakeBuffer(std::uint64_t size, BufferAccessMask access) final;

  std::shared_ptr<hal::Arena> MakeArena(std::uint64_t size, BufferAccessMask access) final;

 private:
  std::shared_ptr<DeviceState> device_state_;
};

// SharedArena implementation

std::mutex SharedArena::svm_mu;

SharedArena::SharedArena(const std::shared_ptr<DeviceState>& device_state, std::uint64_t size)
    : device_state_{device_state}, size_{size} {
  std::lock_guard<std::mutex> lock{svm_mu};
  base_ = clSVMAlloc(device_state_->cl_ctx().get(), CL_MEM_SVM_FINE_GRAIN_BUFFER, size, 0);
  if (!base_) {
    throw error::ResourceExhausted{"Unable to allocate SVM memory"};
  }
}

SharedArena::~SharedArena() {
  if (base_) {
    std::lock_guard<std::mutex> lock{svm_mu};
    clSVMFree(device_state_->cl_ctx().get(), base_);
  }
}

std::shared_ptr<hal::Buffer> SharedArena::MakeBuffer(std::uint64_t offset, std::uint64_t size) {
  if (size_ < offset || size_ < size || size_ < (offset + size)) {
    throw error::OutOfRange{"Requesting memory outside arena bounds"};
  }

  return std::make_shared<SharedBuffer>(shared_from_this(), static_cast<char*>(base_) + offset, size);
}

// SharedBuffer implementation

SharedBuffer::SharedBuffer(std::shared_ptr<SharedArena> arena, void* base, std::uint64_t size)
    : Buffer{arena->device_state()->cl_ctx(), size}, arena_{std::move(arena)}, base_{base} {}

void SharedBuffer::SetKernelArg(const CLObj<cl_kernel>& kernel, std::size_t index) {
  Err::Check(clSetKernelArgSVMPointer(kernel.get(), index, base_), "Unable to set a kernel SVM pointer");
}

boost::future<void*> SharedBuffer::MapCurrent(const std::vector<std::shared_ptr<hal::Event>>& deps) {
  VLOG(4) << "OCL SharedBuffer MapCurrent: waiting this: " << this;
  return Event::WaitFor(deps, arena_->device_state())
      .then([this, base = base_](boost::shared_future<std::vector<std::shared_ptr<hal::Result>>> f) {
        VLOG(4) << "OCL SharedBuffer MapCurrent: complete this: " << this;
        f.get();
        return base;
      });
}

boost::future<void*> SharedBuffer::MapDiscard(const std::vector<std::shared_ptr<hal::Event>>& deps) {
  // We need to wait for the dependencies to resolve; once that's happened, we might as
  // well map the current memory.
  return MapCurrent(deps);
}

std::shared_ptr<hal::Event> SharedBuffer::Unmap(const context::Context& ctx) {
  return std::make_shared<Event>(ctx, arena_->device_state(), CLObj<cl_event>(),
                                 arena_->device_state()->cl_normal_queue());
}

// SharedMemory implementation

SharedMemory::SharedMemory(const std::shared_ptr<DeviceState>& device_state) : device_state_{device_state} {}

std::shared_ptr<hal::Buffer> SharedMemory::MakeBuffer(std::uint64_t size, BufferAccessMask access) {
  return MakeArena(size, access)->MakeBuffer(0, size);
}

std::shared_ptr<hal::Arena> SharedMemory::MakeArena(std::uint64_t size, BufferAccessMask /* access */) {
  return std::make_shared<SharedArena>(device_state_, size);
}

}  // namespace

// Implements Executor::InitSharedMemory on systems that support the
// shared memory OpenCL APIs, by enabling shared memory if the
// underlying hardware supports it.
void Executor::InitSharedMemory() {
  if (!device_state_->info().host_unified_memory()) {
    return;
  }

  for (auto cap : device_state_->info().svm_capability()) {
    if (cap != proto::SvmCapability::FineGrainBuffer) {
      continue;
    }
    VLOG(3) << "Enabling OpenCL fine-grain SVM memory";

    shared_memory_ = compat::make_unique<SharedMemory>(device_state_);
    break;
  }
}

}  // namespace opencl
}  // namespace hal
}  // namespace tile
}  // namespace vertexai
