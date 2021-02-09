// Copyright 2021 Intel Corporation
#include <stdexcept>
#include <vector>

#include "pmlc/rt/level_zero/register.h"
#include "pmlc/rt/runtime.h"
#include "pmlc/rt/runtime_registry.h"
#include "pmlc/util/logging.h"

#include "pmlc/rt/level_zero/level_zero_device.h"

namespace pmlc::rt::level_zero {

class LevelZeroRuntime final : public pmlc::rt::Runtime {
public:
  LevelZeroRuntime() {
    ze_result_t result = ZE_RESULT_NOT_READY;
    try {
      result = zeInit(0);
    } catch (std::exception &e) {
      IVLOG(1, "failed to init level zero driver, result:" << result << " info:"
                                                           << e.what());
      return;
    }
    std::vector<std::pair<ze_driver_handle_t, ze_device_handle_t>>
        supportedDevices = lzu::getSupportedDevices();
    for (auto &target : supportedDevices)
      devices.emplace_back(
          std::make_shared<LevelZeroDevice>(target.first, target.second));
  }

  ~LevelZeroRuntime() {}

  std::size_t deviceCount() const noexcept final { return devices.size(); }
  std::shared_ptr<pmlc::rt::Device> device(std::size_t idx) override {
    if (devices.size() > 0) {
      return devices.at(idx);
    } else {
      IVLOG(1, "Error, no level zero device");
      return NULL;
    }
  }

private:
  std::vector<std::shared_ptr<LevelZeroDevice>> devices;
};

extern void registerSymbols();

void registerRuntime() {
  try {
    registerRuntime("level_zero", std::make_shared<LevelZeroRuntime>());
    registerSymbols();
  } catch (std::exception &ex) {
    IVLOG(1, "Failed to register 'level_zero' runtime: " << ex.what());
  }
}

} // namespace pmlc::rt::level_zero
