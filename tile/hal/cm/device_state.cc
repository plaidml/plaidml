// Copyright 2017-2018 Intel Corporation.

#include "tile/hal/cm/device_state.h"

#include <string>
#include <utility>
#include <vector>

#include <boost/filesystem.hpp>

#include "base/util/env.h"
#include "base/util/error.h"
#include "base/util/file.h"
#include "tile/hal/cm/err.h"
#include "tile/hal/cm/runtime.h"
#include "tile/hal/util/selector.h"

namespace fs = boost::filesystem;

namespace vertexai {
namespace tile {
namespace hal {
namespace cm {

DeviceState::DeviceState(const context::Context& ctx, proto::DeviceInfo dinfo)
    : info_{std::move(dinfo)}, clock_{}, id_{ctx.activity_id()} {}

DeviceState::~DeviceState() {
  RecoverEnv();
  if (pCmDev_) cm_result_check(::DestroyCmDevice(pCmDev_));
}

void DeviceState::Initialize() {
  ConfigEnv();
  MakeDevice();
  MakeQueue();
}

void DeviceState::MakeDevice() {
  UINT version = 0;
  cm_result_check(::CreateCmDevice(pCmDev_, version));
  if (version < CM_1_0) {
    throw std::runtime_error(std::string("The runtime API version is later than runtime DLL version "));
  }
}

void DeviceState::MakeQueue() {
  cm_result_check(pCmDev_->InitPrintBuffer());
  cm_result_check(pCmDev_->CreateQueue(pCmQueue_));
}

void DeviceState::FlushCommandQueue() { cm_result_check(pCmDev_->FlushPrintBuffer()); }

void DeviceState::ConfigEnv() {
  shell_env_ = std::make_unique<ShellEnv>();

  shell_env_->libva_driver_name_ = env::Get("LIBVA_DRIVER_NAME");
  env::Set("LIBVA_DRIVER_NAME", "iHD");

  shell_env_->use_stripe_ = env::Get("USE_STRIPE");
  env::Set("USE_STRIPE", "1");

  shell_env_->plaidml_prohibit_winograd_ = env::Get("PLAIDML_PROHIBIT_WINOGRAD");
  env::Set("PLAIDML_PROHIBIT_WINOGRAD", "1");

  auto cm_cache = env::Get("PLAIDML_CM_CACHE");
  if (cm_cache.length() == 0) {
    std::cout << "Please set PLAIDML_CM_CACHE (full path):" << std::endl;
    getline(std::cin, cm_cache);
    std::cout << "Using PLAIDML_CM_CACHE=" << cm_cache << std::endl;
    env::Set("PLAIDML_CM_CACHE", cm_cache);
  }

  auto cm_root = env::Get("CM_ROOT");
  if (cm_root.length() == 0) {
    std::cout << "Please set CM_ROOT:" << std::endl;
    getline(std::cin, cm_root);
    std::cout << "Using CM_ROOT=" << cm_root << std::endl;
    env::Set("CM_ROOT", cm_root);
  }
}

void DeviceState::RecoverEnv() {
  env::Set("LIBVA_DRIVER_NAME", shell_env_->libva_driver_name_);
  env::Set("USE_STRIPE", shell_env_->use_stripe_);
  env::Set("PLAIDML_PROHIBIT_WINOGRAD", shell_env_->plaidml_prohibit_winograd_);
}

}  // namespace cm
}  // namespace hal
}  // namespace tile
}  // namespace vertexai
