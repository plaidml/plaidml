// Copyright 2020, Intel Corporation

#pragma once

#include "plaidml/edsl/edsl.h"

namespace networks::oplib {

plaidml::edsl::Program buildResnet50(int64_t batch_size = 1);

}  // namespace networks::oplib
