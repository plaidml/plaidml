// Copyright 2017-2018 Intel Corporation.

#pragma once

#include <memory>
#include <vector>

#include "tile/base/hal.h"

namespace vertexai {
namespace tile {
namespace hal {
namespace cpu {

class Executor : public hal::Executor {
 public:
  Executor();

  const hal::proto::HardwareInfo& info() final { return info_; }

  Memory* device_memory() final { return memory_.get(); }

  Memory* shared_memory() final { return memory_.get(); }

  bool is_synchronous() const final { return false; }

  std::shared_ptr<hal::Event> Copy(const context::Context& ctx, const std::shared_ptr<hal::Buffer>& from,
                                   std::size_t from_offset, const std::shared_ptr<hal::Buffer>& to,
                                   std::size_t to_offset, std::size_t length,
                                   const std::vector<std::shared_ptr<hal::Event>>& dependencies) final;

  boost::future<std::unique_ptr<Executable>> Prepare(Library* library) final;

  boost::future<std::vector<std::shared_ptr<hal::Result>>> WaitFor(
      const std::vector<std::shared_ptr<hal::Event>>& events) final;

  void Flush() final {
    // NOP
  }

 private:
  const hal::proto::HardwareInfo info_;
  std::unique_ptr<Memory> memory_;
};

}  // namespace cpu
}  // namespace hal
}  // namespace tile
}  // namespace vertexai
