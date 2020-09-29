// Copyright 2020 Intel Corporation
#pragma once

#include <vector>

#define CL_HPP_ENABLE_EXCEPTIONS
#define CL_HPP_MINIMUM_OPENCL_VERSION 210
#define CL_HPP_TARGET_OPENCL_VERSION 210
#include "CL/cl2.hpp"

namespace pmlc::rt::opencl {

std::vector<cl::Device> getSupportedDevices();

} // namespace pmlc::rt::opencl
