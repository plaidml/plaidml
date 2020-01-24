// Copyright 2020 Intel Corporation

#pragma once

#include <memory>

namespace mlir {
class FuncOp;
class FloatType;
class IntegerType;
template <typename T>
class OpPassBase;
}  // namespace mlir

namespace pmlc::dialect::tile {

class ContractionOp;

std::unique_ptr<mlir::OpPassBase<mlir::FuncOp>> createComputeBoundsPass();

std::unique_ptr<mlir::OpPassBase<mlir::FuncOp>> createConstantTypesPass(mlir::FloatType floatx, mlir::IntegerType intx);

}  // namespace pmlc::dialect::tile
