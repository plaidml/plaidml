// Copyright 2020 Intel Corporation

#pragma once

#include <memory>
#include <string>
#include <vector>

#include "mlir/Support/LLVM.h"
#include "pmlc/rt/runtime.h"

namespace pmlc::rt {

// N.B. These functions are NOT synchronized.  It is the caller's responsibility
// to ensure that other components are not concurrently accessing the system
// global runtime map -- e.g. adding runtimes.

std::shared_ptr<Device> getDevice(mlir::StringRef deviceID);
std::vector<std::string> getDeviceIDs();

} // namespace pmlc::rt
