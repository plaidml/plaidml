// Copyright 2020 Intel Corporation

#include <iostream>
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
    std::cout << "failed to init level zero driver, result:" << result
              << " info:" << e.what() << std::endl;
    return -1;
  }
  std::vector<ze_driver_handle_t> drivers;
  try {
    pmlc::rt::level_zero::lzu::get_all_driver_handles();
  } catch (std::exception &e) {
    std::cout << "inited level zero, but failed to get driver handles:"
              << e.what() << std::endl;
    return -2;
  }
  std::vector<std::pair<ze_driver_handle_t, ze_device_handle_t>>
      supportedDevices = pmlc::rt::level_zero::lzu::getSupportedDevices();
  if (supportedDevices.empty()) {
    std::cout << "No supported level zero devices available" << std::endl;
    return -3;
  }

  std::cout << "Available level zero devices: " << supportedDevices.size()
            << std::endl;
  for (auto &target : supportedDevices) {
    IVLOG(1, "  " << pmlc::rt::level_zero::lzu::get_device_properties(
                         target.second)
                         .name);
  }
  return 0;
#else
  std::cout << "LevelZero is not opened on this platform" << std::endl;
  return 0;
#endif
}
