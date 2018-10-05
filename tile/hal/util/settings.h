// Copyright 2017-2018 Intel Corporation.

#pragma once

#include "tile/lang/generate.h"
#include "tile/proto/hal.pb.h"

namespace vertexai {
namespace tile {
namespace hal {
namespace settings {

// Validates the supplied hardware settings (making sure values are within reasonable limits), throwing an exception if
// there's a problem.
void Validate(const proto::HardwareSettings& settings);

// Translates a hal::proto::HardwareSettings to a lang::HardwareSettings.
// N.B. This does *not* perform validation.
//
// TODO: Consider replacing lang::HardwareSettings with hal::proto::HardwareSettings, since
// they basically contain the same information.
lang::HardwareSettings ToHardwareSettings(const proto::HardwareSettings& settings);

}  // namespace settings
}  // namespace hal
}  // namespace tile
}  // namespace vertexai
