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

std::unique_ptr<mlir::Pass> createComputeBoundsPass();

std::unique_ptr<mlir::Pass> createConstantTypesPass(DataType floatx,
                                                    DataType intx);

struct PaddingPass : public mlir::FunctionPass<PaddingPass> {
  void runOnFunction() final;
};

std::unique_ptr<mlir::Pass> createPaddingPass();

} // namespace pmlc::dialect::tile
