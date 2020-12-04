// Copyright 2020 Intel Corporation

#include <iostream>
#include <vector>

#include "pmlc/rt/level_zero/level_zero_utils.h"

int main(int argc, char **argv) {
  std::vector<ze_driver_handle_t> drivers;
  try {
    drivers = lzt::get_all_driver_handles();
  } catch (std::exception &e) {
    std::cout << e.what() << std::endl;
    throw;
  }
  std::vector<ze_device_handle_t> supportedDevices =
      pmlc::rt::level_zero::getSupportedDevices();
  if (supportedDevices.empty()) {
    std::cout << "No supported level zero devices available" << std::endl;
    return 0;
  }

  std::cout << "Available level zero devices: " << std::endl;
  for (ze_device_handle_t &device : supportedDevices) {
    std::cout << "  " << lzt::get_device_properties(device).name << std::endl;
  }

  return 0;
}
