// Copyright 2020 Intel Corporation

#include "pmlc/all_dialects.h"

#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/Math/IR/Math.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/OpenMP/OpenMPDialect.h"
#include "mlir/Dialect/SCF/SCF.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/Dialect/Vector/IR/VectorOps.h"
#include "mlir/IR/Dialect.h"
#include "mlir/Dialect/Arithmetic/IR/Arithmetic.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"

#include "mlir-hlo/Dialect/mhlo/IR/chlo_ops.h"
#include "mlir-hlo/Dialect/mhlo/IR/hlo_ops.h"

#include "pmlc/dialect/layer/ir/ops.h"
#include "pmlc/dialect/linalgx/ir/ops.h"
#include "pmlc/dialect/pml/ir/dialect.h"
#include "pmlc/dialect/pxa/ir/ops.h"
#include "pmlc/dialect/stdx/ir/ops.h"
#include "pmlc/dialect/tile/ir/ops.h"
#include "pmlc/dialect/xsmm/ir/ops.h"

using namespace mlir; // NOLINT [build/namespaces]

// Add all the MLIR dialects to the provided registry.
void registerAllDialects(DialectRegistry &registry) {
  registry.insert<AffineDialect,                          //
                  LLVM::LLVMDialect,                      //
                  func::FuncDialect,                      //
                  arith::ArithmeticDialect,               //
                  linalg::LinalgDialect,                  //
                  math::MathDialect,                      //
                  memref::MemRefDialect,                  //
                  omp::OpenMPDialect,                     //
                  scf::SCFDialect,                        //
                  tensor::TensorDialect,                  //
                  vector::VectorDialect,                  //
                  mlir::mhlo::MhloDialect,                //
                  pmlc::dialect::layer::LayerDialect,     //
                  pmlc::dialect::linalgx::LinalgXDialect, //
                  pmlc::dialect::pml::PMLDialect,         //
                  pmlc::dialect::pxa::PXADialect,         //
                  pmlc::dialect::stdx::StdXDialect,       //
                  pmlc::dialect::tile::TileDialect,       //
                  arith::ArithmeticDialect,               //
                  pmlc::dialect::xsmm::XSMMDialect>();
}
