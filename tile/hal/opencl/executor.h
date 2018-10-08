// Copyright 2017-2018 Intel Corporation.

#pragma once

#include <memory>
#include <vector>

#include "tile/base/hal.h"
#include "tile/hal/opencl/device_state.h"

namespace vertexai {
namespace tile {
namespace hal {
namespace opencl {

class Executor : public hal::Executor {
 public:
  explicit Executor(const std::shared_ptr<DeviceState>& device_state);

  const hal::proto::HardwareInfo& info() final { return info_; }

  Memory* device_memory() final { return device_memory_.get(); }

  Memory* shared_memory() final { return shared_memory_.get(); }

  bool is_synchronous() const final {
    return !(device_state_->cl_normal_queue().props & CL_QUEUE_OUT_OF_ORDER_EXEC_MODE_ENABLE);
  }

  std::shared_ptr<hal::Event> Copy(const context::Context& ctx, const std::shared_ptr<hal::Buffer>& from,
                                   std::size_t from_offset, const std::shared_ptr<hal::Buffer>& to,
                                   std::size_t to_offset, std::size_t length,
                                   const std::vector<std::shared_ptr<hal::Event>>& dependencies) final;

  boost::future<std::unique_ptr<hal::Executable>> Prepare(Library* library) final;

  boost::future<std::vector<std::shared_ptr<hal::Result>>> WaitFor(
      const std::vector<std::shared_ptr<hal::Event>>& events) final;

  void Flush() final;

 private:
  void InitSharedMemory();

  const std::shared_ptr<DeviceState> device_state_;
  const hal::proto::HardwareInfo info_;
  std::unique_ptr<Memory> device_memory_;
  std::unique_ptr<Memory> shared_memory_;
};

}  // namespace opencl
}  // namespace hal
}  // namespace tile
}  // namespace vertexai
