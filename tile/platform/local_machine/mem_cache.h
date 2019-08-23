// Copyright 2017-2018 Intel Corporation.

#pragma once

#include <cstdint>
#include <memory>
#include <mutex>
#include <stack>
#include <unordered_map>

#include "tile/base/hal.h"
#include "tile/platform/local_machine/devinfo.h"

namespace vertexai {
namespace tile {
namespace local_machine {

// Caches device memory allocations.
class MemCache {
 public:
  MemCache(const std::shared_ptr<DevInfo>& dev_info, hal::Memory* source);
  std::shared_ptr<hal::Buffer> TryAlloc(std::size_t size);
  void Free(std::size_t size, std::shared_ptr<hal::Buffer>);
  void Cleanup(size_t new_size);

 private:
  size_t total_;
  size_t mem_limit_;
  std::mutex mu_;
  hal::Memory* source_;
  std::unordered_map<std::size_t, std::stack<std::shared_ptr<hal::Buffer>>> mem_;
};

}  // namespace local_machine
}  // namespace tile
}  // namespace vertexai
