// Copyright 2020 Intel Corporation

#include <vector>
#ifdef PML_ENABLE_LEVEL_ZERO
#include "pmlc/rt/level_zero/level_zero_utils.h"
#endif
#include "pmlc/util/logging.h"

int main(int argc, char **argv) {
#ifdef PML_ENABLE_LEVEL_ZERO
  ze_result_t result = ZE_RESULT_NOT_READY;
  try {
    result = zeInit(0);
  } catch (std::exception &e) {
    IVLOG(1, "failed to init level zero driver, result:" << result << " info:"
                                                         << e.what());
    return -1;
  }
  std::vector<ze_driver_handle_t> drivers;
  try {
    pmlc::rt::level_zero::lzu::get_all_driver_handles();
  } catch (std::exception &e) {
    IVLOG(1, e.what());
    return -1;
  }
  std::vector<std::pair<ze_driver_handle_t, ze_device_handle_t>>
      supportedDevices = pmlc::rt::level_zero::lzu::getSupportedDevices();
  if (supportedDevices.empty()) {
    IVLOG(1, "No supported level zero devices available");
    return 0;
  }

  IVLOG(1, "Available level zero devices: " << supportedDevices.size());
  for (auto &target : supportedDevices) {
    IVLOG(1, "  " << pmlc::rt::level_zero::lzu::get_device_properties(
                         target.second)
                         .name);
  }
  return 0;
#else
  IVLOG(1, "LevelZero is not opened on this platform");
  return -1;
#endif
}
