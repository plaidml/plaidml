// Copyright 2020, Intel Corporation

#pragma once

#include <memory>
#include <stdexcept>
#include <utility>

#include "pmlc/rt/executable.h"

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

  virtual ~Device() = default;

  virtual std::unique_ptr<Executable>
  compile(const std::shared_ptr<pmlc::compiler::Program> &program,
          llvm::ArrayRef<void *> bufptrs) = 0;

private:
  static std::shared_ptr<Device> currentUntyped();
};

} // namespace pmlc::rt
