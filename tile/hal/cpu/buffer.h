// Copyright 2017-2018 Intel Corporation.

#pragma once

#include <memory>
#include <vector>

#include "tile/base/hal.h"
#include "tile/hal/cpu/arena.h"

namespace vertexai {
namespace tile {
namespace hal {
namespace cpu {

class Buffer : public hal::Buffer {
 public:
  Buffer(std::shared_ptr<Arena> arena, void* base, std::uint64_t size);

  boost::future<void*> MapCurrent(const std::vector<std::shared_ptr<hal::Event>>& deps) final;
  boost::future<void*> MapDiscard(const std::vector<std::shared_ptr<hal::Event>>& deps) final;
  std::shared_ptr<hal::Event> Unmap(const context::Context& ctx) final;

  static std::shared_ptr<Buffer> Downcast(const std::shared_ptr<hal::Buffer>& buffer);

  void* base() const { return base_; }
  std::uint64_t size() const { return size_; }

 private:
  boost::future<void*> Sync(const std::vector<std::shared_ptr<hal::Event>>& deps);

  const std::uint64_t size_;
  void* base_ = nullptr;
  std::shared_ptr<Arena> arena_;
};

}  // namespace cpu
}  // namespace hal
}  // namespace tile
}  // namespace vertexai
