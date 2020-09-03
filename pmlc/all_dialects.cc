// Copyright 2020 Intel Corporation

#include "pmlc/all_dialects.h"

#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/Dialect/GPU/GPUDialect.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/Dialect/Linalg/IR/LinalgOps.h"
#include "mlir/Dialect/OpenMP/OpenMPDialect.h"
#include "mlir/Dialect/Quant/QuantOps.h"
#include "mlir/Dialect/SCF/SCF.h"
#include "mlir/Dialect/SPIRV/SPIRVDialect.h"
#include "mlir/Dialect/StandardOps/IR/Ops.h"
#include "mlir/Dialect/Vector/VectorOps.h"
#include "mlir/IR/Dialect.h"

#include "pmlc/dialect/comp/ir/dialect.h"
#include "pmlc/dialect/eltwise/ir/ops.h"
#include "pmlc/dialect/pxa/ir/ops.h"
#include "pmlc/dialect/stdx/ir/ops.h"
#include "pmlc/dialect/tile/ir/ops.h"
#include "pmlc/dialect/xsmm/ir/ops.h"

using namespace mlir; // NOLINT [build/namespaces]

// Add all the MLIR dialects to the provided registry.
void registerAllDialects(DialectRegistry &registry) {
  registry.insert<AffineDialect,                          //
                  gpu::GPUDialect,                        //
                  LLVM::LLVMDialect,                      //
                  linalg::LinalgDialect,                  //
                  scf::SCFDialect,                        //
                  omp::OpenMPDialect,                     //
                  spirv::SPIRVDialect,                    //
                  StandardOpsDialect,                     //
                  vector::VectorDialect,                  //
                  pmlc::dialect::comp::COMPDialect,       //
                  pmlc::dialect::eltwise::EltwiseDialect, //
                  pmlc::dialect::pxa::PXADialect,         //
                  pmlc::dialect::stdx::StdXDialect,       //
                  pmlc::dialect::tile::TileDialect,       //
                  pmlc::dialect::xsmm::XSMMDialect>();
}

// This function should be called before creating any MLIRContext if one expect
// all the possible dialects to be made available to the context automatically.
void registerAllDialects() {
  static bool initOnce =
      ([]() { registerAllDialects(getGlobalDialectRegistry()); }(), true);
  (void)initOnce;
}
