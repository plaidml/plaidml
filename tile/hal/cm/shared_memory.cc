// Copyright 2017-2018 Intel Corporation.

#include <cstdint>
#include <mutex>
#include <vector>

#include "base/util/compat.h"
#include "base/util/error.h"
#include "base/util/logging.h"
#include "tile/base/hal.h"
#include "tile/hal/cm/buffer.h"
#include "tile/hal/cm/device_state.h"
#include "tile/hal/cm/event.h"
#include "tile/hal/cm/executor.h"

namespace vertexai {
namespace tile {
namespace hal {
namespace cm {

namespace {

// Class declarations

class SharedArena final : public Arena, public std::enable_shared_from_this<SharedArena> {
 public:
  SharedArena(const std::shared_ptr<DeviceState>& device_state, std::uint64_t size);
  virtual ~SharedArena();

  std::shared_ptr<hal::Buffer> MakeBuffer(std::uint64_t offset, std::uint64_t size) final;

  std::shared_ptr<DeviceState> device_state() const { return device_state_; }

 private:
  static std::mutex svm_mu;
  std::shared_ptr<DeviceState> device_state_;
  void* base_ = nullptr;
  std::uint64_t size_;
};

class SharedBuffer final : public Buffer {
 public:
  SharedBuffer(std::shared_ptr<SharedArena> arena, void* base, std::uint64_t size);

  void SetKernelArg(CmKernel* kernel, std::size_t index) final;

  boost::future<void*> MapCurrent(const std::vector<std::shared_ptr<hal::Event>>& deps) final;
  boost::future<void*> MapDiscard(const std::vector<std::shared_ptr<hal::Event>>& deps) final;
  std::shared_ptr<hal::Event> Unmap(const context::Context& ctx) final;

  void* base() const final { return base_; }

 private:
  std::shared_ptr<SharedArena> arena_;
  void* base_ = nullptr;
};

class SharedMemory final : public Memory {
 public:
  explicit SharedMemory(const std::shared_ptr<DeviceState>& device_state);

  std::uint64_t size_goal() const final { return 16 * std::giga::num; }

  BufferAccessMask AllowedAccesses() const final { return BufferAccessMask::ALL; }

  std::size_t ArenaBufferAlignment() const final { return device_state_->info().mem_base_addr_align(); }

  std::shared_ptr<hal::Buffer> MakeBuffer(std::uint64_t size, BufferAccessMask access) final;

  std::shared_ptr<hal::Arena> MakeArena(std::uint64_t size, BufferAccessMask access) final;

 private:
  std::shared_ptr<DeviceState> device_state_;
};

std::mutex SharedArena::svm_mu;

SharedArena::SharedArena(const std::shared_ptr<DeviceState>& device_state, std::uint64_t size)
    : device_state_{device_state}, size_{size} {}

SharedArena::~SharedArena() {}

std::shared_ptr<hal::Buffer> SharedArena::MakeBuffer(std::uint64_t offset, std::uint64_t size) {
  if (size_ < offset || size_ < size || size_ < (offset + size)) {
    throw error::OutOfRange{"Requesting memory outside arena bounds"};
  }
  return std::make_shared<SharedBuffer>(shared_from_this(), static_cast<char*>(base_) + offset, size);
}

SharedBuffer::SharedBuffer(std::shared_ptr<SharedArena> arena, void* base, std::uint64_t size)
    : Buffer{size}, arena_{std::move(arena)}, base_{base} {}

void SharedBuffer::SetKernelArg(CmKernel* kernel, std::size_t index) {}

boost::future<void*> SharedBuffer::MapCurrent(const std::vector<std::shared_ptr<hal::Event>>& deps) {
  VLOG(4) << "OCL SharedBuffer MapCurrent: waiting this: " << this;
  return Event::WaitFor(deps, arena_->device_state()).then([
    this, base = base_
  ](boost::shared_future<std::vector<std::shared_ptr<hal::Result>>> f) {
    f.get();
    return base;
  });
}

boost::future<void*> SharedBuffer::MapDiscard(const std::vector<std::shared_ptr<hal::Event>>& deps) {
  return MapCurrent(deps);
}

std::shared_ptr<hal::Event> SharedBuffer::Unmap(const context::Context& ctx) {
  CmEvent* cm_event;
  return std::make_shared<Event>(ctx, arena_->device_state(), cm_event, arena_->device_state()->cmqueue());
}

SharedMemory::SharedMemory(const std::shared_ptr<DeviceState>& device_state) : device_state_{device_state} {}

std::shared_ptr<hal::Buffer> SharedMemory::MakeBuffer(std::uint64_t size, BufferAccessMask access) {
  return MakeArena(size, access)->MakeBuffer(0, size);
}

std::shared_ptr<hal::Arena> SharedMemory::MakeArena(std::uint64_t size, BufferAccessMask /* access */) {
  return std::make_shared<SharedArena>(device_state_, size);
}

}  // namespace

void Executor::InitSharedMemory() {
  if (!device_state_->info().host_unified_memory()) {
    return;
  }

  for (auto cap : device_state_->info().svm_capability()) {
    if (cap != proto::SvmCapability::FineGrainBuffer) {
      continue;
    }

    shared_memory_ = std::make_unique<SharedMemory>(device_state_);
    break;
  }
}

}  // namespace cm
}  // namespace hal
}  // namespace tile
}  // namespace vertexai
