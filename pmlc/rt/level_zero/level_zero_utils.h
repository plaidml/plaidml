// Copyright 2020 Intel Corporation
#pragma once

#include <utility>
#include <vector>

#include "pmlc/rt/level_zero/utils.h"

namespace pmlc::rt::level_zero::lzu {

std::vector<std::pair<ze_driver_handle_t, ze_device_handle_t>>
getSupportedDevices();

} // namespace pmlc::rt::level_zero::lzu
