// Copyright 2020 Intel Corporation

#pragma once

#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/Dialect/GPU/GPUDialect.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/Dialect/Linalg/IR/LinalgOps.h"
#include "mlir/Dialect/LoopOps/LoopOps.h"
#include "mlir/Dialect/OpenMP/OpenMPDialect.h"
#include "mlir/Dialect/Quant/QuantOps.h"
#include "mlir/Dialect/SPIRV/SPIRVDialect.h"
#include "mlir/Dialect/StandardOps/IR/Ops.h"
#include "mlir/Dialect/Vector/VectorOps.h"
#include "mlir/IR/Dialect.h"

#include "pmlc/dialect/eltwise/ir/ops.h"
#include "pmlc/dialect/pxa/ir/ops.h"
#include "pmlc/dialect/stdx/ir/ops.h"
#include "pmlc/dialect/tile/ir/ops.h"
#include "pmlc/dialect/xsmm/ir/ops.h"

using namespace mlir; // NOLINT [build/namespaces]

// This function should be called before creating any MLIRContext if one expect
// all the possible dialects to be made available to the context automatically.
inline void registerAllDialects() {
  static bool init_once = []() {
    // MLIR core
    registerDialect<AffineDialect>();
    registerDialect<gpu::GPUDialect>();
    registerDialect<LLVM::LLVMDialect>();
    registerDialect<linalg::LinalgDialect>();
    registerDialect<loop::LoopOpsDialect>();
    registerDialect<omp::OpenMPDialect>();
    registerDialect<quant::QuantizationDialect>();
    registerDialect<spirv::SPIRVDialect>();
    registerDialect<StandardOpsDialect>();
    registerDialect<vector::VectorDialect>();
    // PMLC
    registerDialect<pmlc::dialect::eltwise::EltwiseDialect>();
    registerDialect<pmlc::dialect::pxa::PXADialect>();
    registerDialect<pmlc::dialect::stdx::StdXDialect>();
    registerDialect<pmlc::dialect::tile::TileDialect>();
    registerDialect<pmlc::dialect::xsmm::XSMMDialect>();
    return true;
  }();
  (void)init_once;
}
