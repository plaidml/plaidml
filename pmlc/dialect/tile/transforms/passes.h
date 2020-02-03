// Copyright 2020 Intel Corporation

#pragma once

#include <memory>

#include "pmlc/util/enums.h"

namespace mlir {
class FuncOp;
class Pass;
template <typename T>
class OpPassBase;
} // namespace mlir

namespace pmlc::dialect::tile {

using DataType = util::DataType;

class ContractionOp;

std::unique_ptr<mlir::OpPassBase<mlir::FuncOp>> createComputeBoundsPass();

std::unique_ptr<mlir::Pass> createConstantTypesPass(DataType floatx,
                                                    DataType intx);

} // namespace pmlc::dialect::tile
