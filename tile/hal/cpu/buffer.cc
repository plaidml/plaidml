// Copyright 2017-2018 Intel Corporation.

#include "tile/hal/cpu/buffer.h"

#include <memory>
#include <utility>

#include "base/util/error.h"
#include "tile/hal/cpu/event.h"

namespace vertexai {
namespace tile {
namespace hal {
namespace cpu {

std::shared_ptr<Buffer> Buffer::Downcast(const std::shared_ptr<hal::Buffer>& buffer) {
  std::shared_ptr<Buffer> buf = std::dynamic_pointer_cast<Buffer>(buffer);
  if (!buf) {
    throw error::InvalidArgument{"Incompatible buffer for Tile device"};
  }
  return buf;
}

Buffer::Buffer(std::shared_ptr<Arena> arena, void* base, std::uint64_t size)
    : size_{size}, base_{base}, arena_{std::move(arena)} {}

boost::future<void*> Buffer::MapCurrent(const std::vector<std::shared_ptr<hal::Event>>& deps) { return Sync(deps); }

boost::future<void*> Buffer::MapDiscard(const std::vector<std::shared_ptr<hal::Event>>& deps) { return Sync(deps); }

boost::future<void*> Buffer::Sync(const std::vector<std::shared_ptr<hal::Event>>& deps) {
  // Return a future which waits for all of the events, then returns the base
  // address for this buffer, which is always "mapped" since it lives in system
  // memory and cannot be unmapped.
  std::vector<boost::shared_future<std::shared_ptr<hal::Result>>> futures;
  for (auto& ev : deps) {
    futures.emplace_back(ev->GetFuture());
  }
  auto alldeps = boost::when_all(futures.begin(), futures.end());
  return alldeps.then([base = base_](decltype(alldeps) f) {
    f.get();
    return base;
  });
}

std::shared_ptr<hal::Event> Buffer::Unmap(const context::Context& ctx) {
  // We can't unmap host memory, so this is a no-op. Return an event which has
  // already occurred, so it returns a ready future.
  auto now = std::chrono::high_resolution_clock::now();
  return std::make_shared<cpu::Event>(boost::make_ready_future(std::shared_ptr<hal::Result>(std::make_shared<Result>(
                                                                   ctx, "tile::hal::cpu::Buffer::Unmap", now, now)))
                                          .share());
}

}  // namespace cpu
}  // namespace hal
}  // namespace tile
}  // namespace vertexai
