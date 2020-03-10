//===- ConvertGPUToVulkanPass.h - GPU to Vulkan conversion pass -*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// The file declares a pass to convert GPU dialect ops to to Vulkan runtime
// calls.
//
//===----------------------------------------------------------------------===//

#pragma once

#include <memory>

#include "mlir/IR/Module.h"
#include "mlir/Support/LLVM.h"

std::unique_ptr<mlir::OpPassBase<mlir::ModuleOp>>
createConvertMultiGpuLaunchFuncsToVulkanCallsPass();
