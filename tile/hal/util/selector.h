// Copyright 2017-2018 Intel Corporation.

#pragma once

#include <cstdint>

#include "tile/proto/hal.pb.h"

namespace vertexai {
namespace tile {
namespace hal {
namespace selector {

// A test for matching hardware against a selector.
bool Match(const proto::HardwareSelector& sel, const proto::HardwareInfo& info, std::uint_fast32_t depth_allowed = 5);

}  // namespace selector
}  // namespace hal
}  // namespace tile
}  // namespace vertexai
