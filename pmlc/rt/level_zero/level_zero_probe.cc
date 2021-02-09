// Copyright 2021 Intel Corporation

#include <iostream>
#include <vector>
#ifdef PML_ENABLE_LEVEL_ZERO
#include "pmlc/rt/level_zero/level_zero_utils.h"
#endif

int main(int argc, char **argv) {
#ifdef PML_ENABLE_LEVEL_ZERO
  ze_result_t result = ZE_RESULT_NOT_READY;
  try {
    result = zeInit(0);
    if (result != ZE_RESULT_SUCCESS) {
      std::cout << "Function zeInit failed with result: "
                << pmlc::rt::level_zero::lzu::to_string(result) << std::endl;
      return -1;
    }
  } catch (std::exception &e) {
    std::cout << "Function zeInit crashed with result: "
              << pmlc::rt::level_zero::lzu::to_string(result)
              << " info: " << e.what() << std::endl;
    return -2;
  }
  std::vector<ze_driver_handle_t> drivers;
  try {
    pmlc::rt::level_zero::lzu::get_all_driver_handles();
  } catch (std::exception &e) {
    std::cout << "Get level zero driver handle crashed with info: " << e.what()
              << std::endl;
    return -3;
  }
  std::vector<std::pair<ze_driver_handle_t, ze_device_handle_t>>
      supportedDevices = pmlc::rt::level_zero::lzu::getSupportedDevices();
  if (supportedDevices.empty()) {
    std::cout << "No supported level zero devices available" << std::endl;
    return -4;
  }

  std::cout << "Available level zero devices count: " << supportedDevices.size()
            << std::endl;
  for (auto &target : supportedDevices) {
    std::cout
        << "Device: "
        << pmlc::rt::level_zero::lzu::get_device_properties(target.second).name
        << std::endl;
  }
  return 0;
#else
  std::cout << "LevelZero is not opened on this platform" << std::endl;
  return -1;
#endif
}
