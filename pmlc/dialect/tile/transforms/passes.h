// Copyright 2020 Intel Corporation

#pragma once

#include "pmlc/util/enums.h"
#include <memory>

namespace mlir {
class FuncOp;
class MLIRContext;
template <typename T>
class OpPassBase;
}  // namespace mlir

namespace pmlc::dialect::tile {

using DataType = util::DataType;

class ContractionOp;
class ConstantTypesPass;

std::unique_ptr<mlir::OpPassBase<mlir::FuncOp>> createComputeBoundsPass();

std::unique_ptr<mlir::Pass> createConstantTypesPass(const DataType& floatx, const DataType& intx);

}  // namespace pmlc::dialect::tile
