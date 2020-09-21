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
// A Factory is typically a small, stateless function, safely instantiated at
// static initialization time to defer Runtime instantiation to a time after
// static initialization is complete.
//
// Typical runtime implementations have no need to explicitly use Factory.  To
// register a DefaultConstructible Runtime at static initialization time, the
// runtime implementation should use RuntimeRegistration, which will
// automatically create and register a Factory.
//
// (Alternatively, if the Runtime is instantiated explicitly after static
// initialization time, the caller may use bindRuntime() to add the runtime to
// the system's global runtime map.)
//
// Factory is invoked on-demand (typically well after static initialization) to
// instantiate a runtime, either as a specific request by the PlaidML caller or
// for general device enumeration.  The registry is responsible for arranging
// runtimes to be instantiated exactly once; Factory implementations should not
// memoize their results.
using Factory = std::function<std::shared_ptr<Runtime>()>;

// Creates a global registration for a factory.
//
// This function does not check the ID for conflicts with other runtimes;
// duplicate checking will occur when initRuntimes() is called.  The caller is
// responsible for arranging for initRuntimes() to be called after this call
// returns, in order to actually instantiate the Runtime.
//
// N.B. This function is NOT synchronized.  It is the caller's responsiblity to
// ensure that other components are not concurrently accessing the system global
// runtime map or registering other Factory instances.
void registerFactory(llvm::StringRef id, Factory factory);

// makeStaticFactory is a template function returning a Factory that
// instantiates an instance of the supplied concrete Runtime class.  The Runtime
// class must be DefaultConstructible.
template <class R>
Factory makeStaticFactory() {
  return Factory{[]() { return std::make_shared<R>(); }};
}

// makeConstantFactory returns a Factory that always returns the given Runtime.
inline Factory makeConstantFactory(std::shared_ptr<Runtime> runtime) {
  return Factory{[runtime = std::move(runtime)]() { return runtime; }};
}

// FactoryRegistration is used to register a PlaidML Runtime Factory at static
// initialization time. Components whose Runtimes are DefaultConstructible are
// encouraged to use RuntimeRegistration instead.
struct FactoryRegistration {
  FactoryRegistration(llvm::StringRef id, Factory factory) {
    registerFactory(id, std::move(factory));
  }
};

// RuntimeRegistration is used to register a PlaidML Runtime class at static
// initialization time.  This is used by runtimes that are linked into the image
// (vs. being dynamically discovered and loaded).
//
// For example:
//
//   RuntimeRegistration<OpenCLRuntime> reg{"opencl"};
//
template <class R>
struct RuntimeRegistration {
  explicit RuntimeRegistration(llvm::StringRef id) {
    registerFactory(id, makeStaticFactory<R>());
  }

  RuntimeRegistration(llvm::StringRef id, std::shared_ptr<R> runtime) {
    registerFactory(
        id, makeConstantFactory(std::shared_ptr<Runtime>{std::move(runtime)}));
  }
};

// registerRuntime directly binds the indicated runtime ID to the supplied
// runtime in the system's global runtime map.  This will throw an exception if
// the new runtime's ID conflicts with an existing ID.
//
// N.B. This function is NOT synchronized.  It is the caller's responsibility to
// ensure that other components are not concurrently accessing the system global
// runtime map -- e.g. looking up device IDs or compiling programs to produce
// executables.
void registerRuntime(llvm::StringRef id, std::shared_ptr<Runtime> runtime);

// initRuntimes processes registered factories and adds them to the system's
// global runtime map.
//
// N.B. This function is NOT synchronized.  It is the caller's responsibility to
// ensure that other components are not concurrently accessing the system global
// runtime map
// -- e.g. looking up device IDs or compiling programs to produce
// executables.
void initRuntimes();

} // namespace pmlc::rt
