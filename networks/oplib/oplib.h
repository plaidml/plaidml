// Copyright 2020, Intel Corporation

#pragma once

#include "plaidml/core/core.h"
#include "plaidml/exec/exec.h"

namespace networks::oplib {

plaidml::Program buildResnet50(int64_t batch_size = 1);

plaidml::exec::Executable createDefaultExecutable(plaidml::Program program);

}  // namespace networks::oplib
