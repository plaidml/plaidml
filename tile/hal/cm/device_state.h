// Copyright 2017-2018 Intel Corporation.

#pragma once

#include <memory>
#include <string>

#include "base/context/context.h"
#include "tile/hal/cm/cm.pb.h"
#include "tile/hal/cm/runtime.h"
#include "tile/lang/generate.h"

namespace vertexai {
namespace tile {
namespace hal {
namespace cm {

class DeviceState {
 public:
  struct ShellEnv {
    std::string libva_driver_name_;
    std::string use_stripe_;
    std::string plaidml_prohibit_winograd_;
  };

  DeviceState(const context::Context& ctx, proto::DeviceInfo dinfo);
  ~DeviceState();

  CmDevice* cmdev() {
    if (pCmDev_ == nullptr) {
      Initialize();
    }
    return pCmDev_;
  }

  CmQueue* cmqueue() const { return pCmQueue_; }
  const proto::DeviceInfo info() const { return info_; }
  const context::Clock& clock() const { return clock_; }
  const context::proto::ActivityID& id() const { return id_; }

  void FlushCommandQueue();

 private:
  void Initialize();
  void MakeDevice();
  void MakeQueue();
  void ConfigEnv();
  void RecoverEnv();

  CmDevice* pCmDev_ = nullptr;
  CmQueue* pCmQueue_;
  std::unique_ptr<ShellEnv> shell_env_;
  const proto::DeviceInfo info_;
  const context::Clock clock_;
  const context::proto::ActivityID id_;
};

}  // namespace cm
}  // namespace hal
}  // namespace tile
}  // namespace vertexai
