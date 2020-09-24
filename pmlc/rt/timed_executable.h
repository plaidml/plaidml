// Copyright 2020 Intel Corporation

#pragma once

#include <memory>

#include "pmlc/rt/executable.h"

namespace pmlc::rt {

// Returns an Executable implementation that times how long it takes to perform
// an inference.
std::unique_ptr<Executable>
makeTimedExecutable(std::unique_ptr<Executable> base);

} // namespace pmlc::rt
