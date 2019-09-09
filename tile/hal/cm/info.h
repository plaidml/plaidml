// Copyright 2017-2018 Intel Corporation.

#pragma once

#include <string>
#include <utility>
#include <vector>

#include <boost/regex.hpp>

#include "tile/hal/cm/cm.pb.h"
#include "tile/proto/hal.pb.h"

namespace vertexai {
namespace tile {
namespace hal {
namespace cm {

hal::proto::HardwareInfo GetHardwareInfo(const proto::DeviceInfo& info);

}  // namespace cm
}  // namespace hal
}  // namespace tile
}  // namespace vertexai
