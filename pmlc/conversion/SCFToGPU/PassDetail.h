//===- PassDetail.h - Conversion Pass class details -------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#pragma once

#include "mlir/Pass/Pass.h"

namespace pmlc::conversion::scf_to_gpu {

namespace gpu {
class GPUModuleOp;
}

#define GEN_PASS_CLASSES
#include "pmlc/conversion/SCFToGPU/Passes.h.inc"

} // namespace pmlc::conversion::scf_to_gpu
