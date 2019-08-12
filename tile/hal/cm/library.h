// Copyright 2017-2018 Intel Corporation.

#pragma once

#include <map>
#include <memory>
#include <string>
#include <vector>

#include "tile/base/hal.h"
#include "tile/hal/cm/device_state.h"
#include "tile/hal/cm/emitcm.h"
#include "tile/hal/cm/runtime.h"
#include "tile/lang/generate.h"

namespace vertexai {
namespace tile {
namespace hal {
namespace cm {

class Library final : public hal::Library {
 public:
  static Library* Downcast(hal::Library* library, std::shared_ptr<DeviceState> device_state);

  Library(std::shared_ptr<DeviceState> device_state, const std::map<std::string, CmProgram*>& program_map,
          const std::map<std::string, std::shared_ptr<Emit>>& emit_map,
          const std::vector<lang::KernelInfo>& kernel_info, std::vector<context::proto::ActivityID> kernel_ids);

  std::map<std::string, std::string> Serialize() final;

  std::shared_ptr<DeviceState> device_state() { return device_state_; }
  const std::map<std::string, CmProgram*>& getProgramMap() const { return program_map_; }
  const std::map<std::string, std::shared_ptr<Emit>>& get_emit_map() const { return emit_map_; }
  const std::vector<lang::KernelInfo>& kernel_info() const { return kernel_info_; }
  const std::vector<context::proto::ActivityID>& kernel_ids() const { return kernel_ids_; }

 private:
  std::shared_ptr<DeviceState> device_state_;
  std::map<std::string, CmProgram*> program_map_;
  std::map<std::string, std::shared_ptr<Emit>> emit_map_;
  std::vector<lang::KernelInfo> kernel_info_;
  std::vector<context::proto::ActivityID> kernel_ids_;
};

}  // namespace cm
}  // namespace hal
}  // namespace tile
}  // namespace vertexai
