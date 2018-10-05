// Copyright 2017-2018 Intel Corporation.

#include "tile/platform/local_machine/mem_cache.h"

#include <utility>

namespace vertexai {
namespace tile {
namespace local_machine {

std::shared_ptr<hal::Buffer> MemCache::TryAlloc(std::size_t size) {
  std::lock_guard<std::mutex> lock{mu_};
  auto& l = mem_[size];
  if (l.size()) {
    std::shared_ptr<hal::Buffer> result{std::move(l.top())};
    l.pop();
    return result;
  }
  return std::shared_ptr<hal::Buffer>{};
}

void MemCache::Free(std::size_t size, std::shared_ptr<hal::Buffer> mem) {
  std::lock_guard<std::mutex> lock{mu_};
  mem_[size].push(std::move(mem));
}

}  // namespace local_machine
}  // namespace tile
}  // namespace vertexai
