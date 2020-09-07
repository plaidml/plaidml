// Copyright 2020 Intel Corporation

#include <iostream>
#include <vector>

#define CL_HPP_ENABLE_EXCEPTIONS
#define CL_HPP_MINIMUM_OPENCL_VERSION 120
#define CL_HPP_TARGET_OPENCL_VERSION 120
#include "CL/cl2.hpp"

int main(int argc, char **argv) {
  std::vector<cl::Platform> platforms;
  try {
    cl::Platform::get(&platforms);
  } catch (const cl::Error &e) {
    if (CL_PLATFORM_NOT_FOUND_KHR == e.err()) {
      std::cout << "No availiable OpenCL platforms" << std::endl;
      return 0;
    }
    throw;
  }
  std::cout << "Availiable platforms and devices:" << std::endl;
  for (auto &p : platforms) {
    std::cout << p.getInfo<CL_PLATFORM_NAME>() << std::endl;
    std::vector<cl::Device> devices;
    p.getDevices(CL_DEVICE_TYPE_ALL, &devices);
    for (auto &d : devices) {
      std::cout << "  " << d.getInfo<CL_DEVICE_NAME>() << std::endl;
    }
  }

  return 0;
}
