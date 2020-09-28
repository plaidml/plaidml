// Copyright 2020 Intel Corporation

#include <iostream>
#include <vector>

#include "pmlc/rt/opencl/opencl_utils.h"

int main(int argc, char **argv) {
  std::vector<cl::Platform> platforms;
  try {
    cl::Platform::get(&platforms);
  } catch (const cl::Error &e) {
    if (CL_PLATFORM_NOT_FOUND_KHR != e.err())
      throw;
  }
  std::vector<cl::Device> supportedDevices =
      pmlc::rt::opencl::getSupportedDevices();
  if (supportedDevices.empty()) {
    std::cout << "No supported OpenCL devices available" << std::endl;
    return 0;
  }

  std::cout << "Available OpenCL devices: " << std::endl;
  for (cl::Device &device : supportedDevices) {
    std::cout << "  " << device.getInfo<CL_DEVICE_NAME>() << std::endl;
  }

  return 0;
}
