// Copyright 2020, Intel Corporation

#pragma once

#include <cstddef>
#include <memory>
#include <stdexcept>
#include <utility>

namespace pmlc::rt {

// Device represents a PlaidML device, capable of evaluating a PlaidML program.
class Device {
public:
  // Returns the current Device, local to the current thread.
  //
  // If there is no current device, or the current device is of an unexpected
  // type, an exception will be thrown.
  //
  // This is used to smuggle the current device through to runtimes that are
  // directly invoked by lowered code, when that lowered code doesn't support
  // the use of a passthrough parameter.
  template <typename T>
  static std::shared_ptr<T> current() {
    std::shared_ptr<Device> dev = currentUntyped();
    if (!dev) {
      throw std::logic_error{"No current device established"};
    }
    std::shared_ptr<T> result = std::dynamic_pointer_cast<T>(std::move(dev));
    if (!result) {
      throw std::runtime_error{"Incompatible target/device combination"};
    }
    return result;
  }

  virtual ~Device() {}

private:
  static std::shared_ptr<Device> currentUntyped();
};

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
