//===- SCFToGPUPass.h - Pass converting loops to GPU kernels ----*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#pragma once

#include <memory>

#include "mlir/Support/LLVM.h"

namespace mlir {
class FuncOp;
template <typename T>
class OperationPass;
class Pass;
} // namespace mlir

namespace pmlc::conversion::scf_to_gpu {

/// Create a pass that converts loop nests into GPU kernels.  It considers
/// top-level affine.for and linalg.for operations as roots of loop nests and
/// converts them to the gpu.launch operations if possible.
///
/// No check on the size of the block or grid, or on the validity of
/// parallelization is performed, it is under the responsibility of the caller
/// to strip-mine the loops and to perform the dependence analysis before
/// calling the conversion.
std::unique_ptr<mlir::OperationPass<mlir::FuncOp>>
createSimpleSCFToGPUPass(unsigned numBlockDims, unsigned numThreadDims);

std::unique_ptr<mlir::OperationPass<mlir::FuncOp>> createSimpleSCFToGPUPass();

} // namespace pmlc::conversion::scf_to_gpu
