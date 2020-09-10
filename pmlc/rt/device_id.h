// Copyright 2020 Intel Corporation

#pragma once

#include <memory>
#include <string>
#include <vector>

#include "pmlc/rt/runtime.h"
#include "llvm/ADT/StringRef.h"

namespace pmlc::rt {

std::shared_ptr<Device> getDevice(llvm::StringRef deviceID);
std::vector<std::string> getDeviceIDs();

} // namespace pmlc::rt
