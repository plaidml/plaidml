// Copyright 2020 Intel Corporation

#include "pmlc/rt/llvm/register.h"
#include "pmlc/rt/opencl/register.h"
#include "pmlc/rt/register.h"
#include "pmlc/rt/runtime_registry.h"
#include "pmlc/rt/vulkan/register.h"

namespace pmlc::rt {

void registerRuntimes() {
  registerRuntime();
  llvm::registerRuntime();
  opencl::registerRuntime();
  vulkan::registerRuntime();
}

} // namespace pmlc::rt
