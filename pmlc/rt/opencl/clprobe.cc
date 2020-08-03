// Copyright 2020 Intel Corporation

#include <iostream>
#include <vector>

#include "CL/cl2.hpp"

int main(int argc, char **argv) {
  std::cout << "Availiable platforms and devices:" << std::endl;
  std::vector<cl::Platform> platforms;
  cl::Platform::get(&platforms);
  for (auto &p : platforms) {
    std::cout << p.getInfo<CL_PLATFORM_NAME>() << std::endl;
    std::vector<cl::Device> devices;
    p.getDevices(CL_DEVICE_TYPE_ALL, &devices);
    for (auto &d : devices) {
      std::cout << "  " << d.getInfo<CL_DEVICE_NAME>() << std::endl;
    }
  }

  return 1;
}
