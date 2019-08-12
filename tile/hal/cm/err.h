// Copyright 2017-2018 Intel Corporation.

#pragma once

#include <string>

#include "base/util/logging.h"
#include "tile/hal/cm/buffer.h"
#include "tile/hal/cm/device_state.h"
#include "tile/hal/cm/runtime.h"

namespace vertexai {
namespace tile {
namespace hal {
namespace cm {

#define cm_result_check(x) checkError((x), __FILE__, __LINE__, true);

const char* getErrorString(int Code);

int checkError(int Code, const char* Path, const int Line, bool Abort);

}  // namespace cm
}  // namespace hal
}  // namespace tile
}  // namespace vertexai
