// Copyright 2020, Intel Corporation

#pragma once

#include <memory>
#include <utility>

#include "llvm/ADT/StringRef.h"

#include "pmlc/rt/runtime.h"

namespace pmlc::rt {

namespace details {

void registerRuntime(llvm::StringRef id, std::unique_ptr<Runtime> runtime);

} // namespace details

// RuntimeRegistration is used to register a PlaidML Runtime class at static
// initialization time (at runtime, prior to main()).
//
// N.B. This requires that an instance of the runtime must be instantiated at
//      static initialization time.  Implementations must perform
//      initializations within init().
//
// For example:
//
//   RuntimeRegistration reg{"opencl", std::make_unique<OpenCLRuntime>()};
//
struct RuntimeRegistration {
  explicit RuntimeRegistration(llvm::StringRef id,
                               std::unique_ptr<Runtime> runtime) {
    details::registerRuntime(id, std::move(runtime));
  }
};

} // namespace pmlc::rt
