// Copyright 2017-2018 Intel Corporation.

#pragma once

#include <memory>
#include <vector>

#include "tile/hal/cm/buffer.h"
#include "tile/hal/cm/device_state.h"
#include "tile/hal/cm/err.h"
#include "tile/hal/cm/runtime.h"

namespace vertexai {
namespace tile {
namespace hal {
namespace cm {

class CMMemBuffer final : public Buffer, public std::enable_shared_from_this<CMMemBuffer> {
 public:
  CMMemBuffer(std::shared_ptr<DeviceState> device_state, std::uint64_t size, CmBufferUP* pCmBuffer, void* base);

  ~CMMemBuffer();

  void SetKernelArg(CmKernel* kernel, std::size_t index) final;

  boost::future<void*> MapCurrent(const std::vector<std::shared_ptr<hal::Event>>& deps) final;
  boost::future<void*> MapDiscard(const std::vector<std::shared_ptr<hal::Event>>& deps) final;
  std::shared_ptr<hal::Event> Unmap(const context::Context& ctx) final;

  CmBufferUP* mem() const final { return pCmBuffer_; }

  void clean_base_() { memset(base_, 0, this->size()); }
  void* getbase() { return base_; }
  void ReleaseDeviceBuffer();

 private:
  static CmBufferUP* MakeMem(std::shared_ptr<DeviceState> device_state, std::uint64_t size);

  std::shared_ptr<DeviceState> device_state_;
  CmBufferUP* pCmBuffer_;
  void* base_;
};

}  // namespace cm
}  // namespace hal
}  // namespace tile
}  // namespace vertexai
