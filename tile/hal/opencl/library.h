// Copyright 2017-2018 Intel Corporation.

#pragma once

#include <memory>
#include <string>
#include <vector>

#include "tile/base/hal.h"
#include "tile/hal/opencl/device_state.h"
#include "tile/hal/opencl/ocl.h"
#include "tile/lang/generate.h"

namespace vertexai {
namespace tile {
namespace hal {
namespace opencl {

class Library final : public hal::Library {
 public:
  static Library* Downcast(hal::Library* library, const std::shared_ptr<DeviceState>& device_state);

  Library(const std::shared_ptr<DeviceState>& device_state, CLObj<cl_program> program,
          const std::vector<lang::KernelInfo>& kernel_info, std::vector<context::proto::ActivityID> kernel_ids);

  std::string Serialize() final;

  const std::shared_ptr<DeviceState>& device_state() const { return device_state_; }
  const CLObj<cl_program>& program() const { return program_; }
  const std::vector<lang::KernelInfo>& kernel_info() const { return kernel_info_; }
  const std::vector<context::proto::ActivityID>& kernel_ids() const { return kernel_ids_; }

 private:
  std::shared_ptr<DeviceState> device_state_;
  CLObj<cl_program> program_;
  std::vector<lang::KernelInfo> kernel_info_;
  std::vector<context::proto::ActivityID> kernel_ids_;
};

}  // namespace opencl
}  // namespace hal
}  // namespace tile
}  // namespace vertexai
