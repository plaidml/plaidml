// Copyright 2020 Intel Corporation

#include "pmlc/rt/llvm/register.h"
#include "pmlc/rt/register.h"
#include "pmlc/rt/runtime_registry.h"

#ifdef PML_ENABLE_OPENCL
#include "pmlc/rt/opencl/register.h"
#endif
#ifdef PML_ENABLE_VULKAN
#include "pmlc/rt/vulkan/register.h"
#endif

namespace pmlc::rt {

void registerRuntimes() {
  registerRuntime();
  llvm::registerRuntime();
#ifdef PML_ENABLE_OPENCL
  opencl::registerRuntime();
#endif
#ifdef PML_ENABLE_VULKAN
  vulkan::registerRuntime();
#endif
}

} // namespace pmlc::rt
