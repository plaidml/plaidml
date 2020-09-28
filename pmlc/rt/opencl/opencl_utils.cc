// Copyright 2020 Intel Corporation
#include "pmlc/rt/opencl/opencl_utils.h"

#include <string>
#include <vector>

namespace pmlc::rt::opencl {

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

static bool isPlatformSupported(const cl::Platform &p) {
  std::string platformVersion = p.getInfo<CL_PLATFORM_VERSION>();
  unsigned versionNum = extractOpenCLVersion(platformVersion);
  if (versionNum < CL_HPP_MINIMUM_OPENCL_VERSION)
    return false;
  return true;
}

std::vector<cl::Device> getSupportedDevices() {
  std::vector<cl::Platform> platforms;
  try {
    cl::Platform::get(&platforms);
  } catch (const cl::Error &e) {
    if (CL_PLATFORM_NOT_FOUND_KHR != e.err())
      throw;
  }
  std::vector<cl::Device> supportedDevices;
  for (auto &p : platforms) {
    if (!isPlatformSupported(p))
      continue;
    std::vector<cl::Device> devices;
    p.getDevices(CL_DEVICE_TYPE_GPU, &devices);
    supportedDevices.insert(supportedDevices.end(), devices.begin(),
                            devices.end());
  }
  return supportedDevices;
}

} // namespace pmlc::rt::opencl
