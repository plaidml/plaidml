// Copyright 2017-2018 Intel Corporation.

#pragma once

#include <cstdint>
#include <memory>
#include <mutex>
#include <stack>
#include <unordered_map>

#include "tile/base/hal.h"

namespace vertexai {
namespace tile {
namespace local_machine {

// Caches device memory allocations.
class MemCache {
 public:
  std::shared_ptr<hal::Buffer> TryAlloc(std::size_t size);
  void Free(std::size_t size, std::shared_ptr<hal::Buffer>);

 private:
  std::mutex mu_;
  std::unordered_map<std::size_t, std::stack<std::shared_ptr<hal::Buffer>>> mem_;
};

}  // namespace local_machine
}  // namespace tile
}  // namespace vertexai
