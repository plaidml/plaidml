// Copyright 2020 Intel Corporation
#include "pmlc/rt/level_zero/level_zero_utils.h"

#include <string>
#include <vector>

namespace pmlc::rt::level_zero {
#if 0
// Returns OpenCL version as decimal number, ie:
// OpenCL 1.2 - 120, OpenCL 2.2 - 220
static unsigned extractOpenCLVersion(const std::string &versionString) {
  unsigned version = 0;
  unsigned idx = 7; // OpenCL<space>
  while (versionString[idx] != '.') {
    version *= 10;
    version += versionString[idx] - '0';
    ++idx;
  }
  ++idx;
  while (versionString[idx] != ' ' && idx < versionString.size()) {
    version *= 10;
    version += versionString[idx] - '0';
    ++idx;
  }
  return version * 10;
}
#endif
#if 0
static bool isPlatformSupported(const cl::Platform &p) {
  std::string platformVersion = p.getInfo<CL_PLATFORM_VERSION>();
  unsigned versionNum = extractOpenCLVersion(platformVersion);
  if (versionNum < CL_HPP_MINIMUM_OPENCL_VERSION)
    return false;
  return true;
}
#endif
std::vector<ze_device_handle_t> getSupportedDevices() {
  std::vector<ze_device_handle_t> supportedDevices;
  for (auto driver : lzt::get_all_driver_handles()) {
    for (auto device : lzt::get_devices(driver))
      supportedDevices.push_back(device);
  }
  return supportedDevices;
}

} // namespace pmlc::rt::level_zero
