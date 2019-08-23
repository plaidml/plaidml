// Copyright 2017-2018 Intel Corporation.

#include "tile/platform/local_machine/mem_cache.h"

#include <utility>

namespace vertexai {
namespace tile {
namespace local_machine {

MemCache::MemCache(const std::shared_ptr<DevInfo>& dev_info, hal::Memory* source) :
  total_{0}, source_{source} {
  mem_limit_ = dev_info->settings.max_global_mem();
}

void MemCache::Cleanup(size_t new_size) {
  while (total_ + new_size > mem_limit_) {
    // Out of device memory, clean up cache
    while (mem_.size() > 0 && mem_.begin()->second.empty()) {
      mem_.erase(mem_.begin());
    }
    if (mem_.empty()) {
      break;
    }
    total_ -= mem_.begin()->first;
    mem_.begin()->second.pop();
  }
}

std::shared_ptr<hal::Buffer> MemCache::TryAlloc(std::size_t size) {
  std::lock_guard<std::mutex> lock{mu_};
  auto& l = mem_[size];
  if (l.size()) {
    std::shared_ptr<hal::Buffer> result{std::move(l.top())};
    l.pop();
    return result;
  }
  Cleanup(size);
  total_ += size;
  auto buffer = source_->MakeBuffer(size, hal::BufferAccessMask::DEVICE_RW);
  return std::shared_ptr<hal::Buffer>{std::move(buffer)};
}

void MemCache::Free(std::size_t size, std::shared_ptr<hal::Buffer> mem) {
  std::lock_guard<std::mutex> lock{mu_};
  Cleanup(size);
  total_ += size;
  mem_[size].push(std::move(mem));
}

}  // namespace local_machine
}  // namespace tile
}  // namespace vertexai
