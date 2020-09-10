// Copyright 2020 Intel Corporation

#pragma once

#include <memory>
#include <regex>
#include <string>
#include <unordered_map>

#include "llvm/ADT/StringMap.h"

#include "pmlc/rt/runtime.h"

namespace pmlc::rt {

// ScopedCurrentDevice sets the current thread's Device pointer.  This is read
// by Device::current().
class ScopedCurrentDevice {
public:
  explicit ScopedCurrentDevice(std::shared_ptr<Device> device);
  ScopedCurrentDevice(const ScopedCurrentDevice &) = delete;
  ~ScopedCurrentDevice();
};

// getSymbolMap returns the map of symbols registered by components within the
// current process.
llvm::StringMap<void *> &getSymbolMap();

// getDeviceIDRegex returns the regular expression used to match device
// identifiers.
//
// When matching a device identifier, match[1] will be the runtime identifier,
// and match[3] will be the device index within that runtime.
const std::regex &getDeviceIDRegex();

// getRuntimeMap returns the map of runtimes available to the current process.
// N.B. This is mutable at static initialization time, but should be treated as
// immutable after main() has been called; there is no associated lock.
std::unordered_map<std::string, std::unique_ptr<Runtime>> &getRuntimeMap();

} // namespace pmlc::rt
