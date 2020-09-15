// Copyright 2020, Intel Corporation

#pragma once

#include <cstddef>
#include <memory>

#include "pmlc/rt/device.h"

namespace pmlc::rt {

// Runtime represents a particular runtime implementation.
//
// A runtime implementation is a component that understands how to use a
// particular API (e.g. OpenCL, Metal, Vulkan, &c) to access a device capable of
// evaluating a PlaidML program.  These are typically single-instanced within a
// process, constructed once as needed (typically well after process
// initialization) and then held onto by the global runtime registry.
class Runtime {
public:
  virtual ~Runtime() {}

  // Returns the devices supported by this Runtime.
  virtual std::size_t deviceCount() const noexcept = 0;
  virtual std::shared_ptr<Device> device(std::size_t idx) = 0;
};

} // namespace pmlc::rt
