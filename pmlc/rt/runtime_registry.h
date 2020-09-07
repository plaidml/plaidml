// Copyright 2020, Intel Corporation

#pragma once

#include <functional>
#include <memory>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>

#include "llvm/ADT/StringRef.h"

#include "pmlc/rt/runtime.h"

namespace pmlc::rt {

// Factory is used to instantiate a Runtime.
//
// Factory is invoked on-demand (typically well after process initialization) to
// instantiate a runtime, either as a specific request by the PlaidML caller or
// for general device enumeration.  The registry is responsible for arranging
// runtimes to be instantiated exactly once; Factory implementations do not need
// to memoize their results.
using Factory = std::function<std::shared_ptr<Runtime>()>;

// Loaders instantiate Factories.
//
// PlaidML Loaders are (typically) simple, stateless functions, automatically
// registered via LoaderRegistration when their containing library is loaded
// into the process.
//
// Loaders are invoked on-demand (typically well after process initialization)
// to determine the runtime factories they're able to provide.
using Loader = std::function<std::unordered_map<std::string, Factory>()>;

namespace details {

// Creates a global registration for a loader.
void registerLoader(llvm::StringRef id, Loader loader);

} // namespace details

// makeStaticLoader is a template function returning a Loader that instantiates
// an instance of the supplied concrete Runtime class.  The Runtime class must
// be DefaultConstructible.
template <class R>
Loader makeStaticLoader(llvm::StringRef id) {
  return Loader{[savedId = std::string{id}]() {
    return std::unordered_map<std::string, Factory>{
        {savedId, Factory{[]() { return std::make_shared<R>(); }}}};
  }};
}

// LoaderRegistration is used to register a PlaidML Loader at image load time.
//
// For example:
//
//   LoaderRegistration reg{makeStaticLoader<OpenCLLoader>("opencl")};
//
struct LoaderRegistration {
  LoaderRegistration(llvm::StringRef id, Loader loader) {
    details::registerLoader(id, std::move(loader));
  }
};

// RuntimeRegistration is used to register a PlaidML Runtime class at image load
// time.  This is used by runtimes that are linked into the image (vs. being
// dynamically discovered and loaded).
//
// For example:
//
//   RuntimeRegistration<OpenCLRuntime> reg{"opencl"};
//
template <class R>
struct RuntimeRegistration {
  explicit RuntimeRegistration(llvm::StringRef id) {
    details::registerLoader(id, makeStaticLoader<R>(id));
  }
};

} // namespace pmlc::rt
