// Copyright 2017-2018 Intel Corporation.

#pragma once

#include <memory>
#include <vector>

#include "tile/base/hal.h"
#include "tile/hal/cm/runtime.h"

namespace vertexai {
namespace tile {
namespace hal {
namespace cm {

class Buffer : public hal::Buffer {
 public:
  static std::shared_ptr<Buffer> Downcast(const std::shared_ptr<hal::Buffer>& buffer);
  static Buffer* Downcast(hal::Buffer* buffer);

  virtual void SetKernelArg(CmKernel* kernel, std::size_t index) = 0;
  virtual void ReleaseDeviceBuffer() {}
  virtual void* base() const { return nullptr; }
  virtual CmBufferUP* mem() const { return nullptr; }
  std::uint64_t size() const { return size_; }

 protected:
  explicit Buffer(std::uint64_t size);

 private:
  const std::uint64_t size_;
};

}  // namespace cm
}  // namespace hal
}  // namespace tile
}  // namespace vertexai
