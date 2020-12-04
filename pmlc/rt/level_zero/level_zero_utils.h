// Copyright 2020 Intel Corporation
#pragma once

#include <level_zero/ze_api.h>

#include <array>
#include <vector>

//#define CL_HPP_ENABLE_EXCEPTIONS
//#define CL_HPP_MINIMUM_OPENCL_VERSION 210
//#define CL_HPP_TARGET_OPENCL_VERSION 210

#include "pmlc/rt/level_zero/utils/include/test_harness/test_harness.hpp"
#include "pmlc/rt/level_zero/utils/include/utils/logging.hpp"
#include "pmlc/rt/level_zero/utils/include/utils/utils.hpp"

namespace pmlc::rt::level_zero {

std::vector<ze_device_handle_t> getSupportedDevices();

} // namespace pmlc::rt::level_zero
