// Copyright 2020 Intel Corporation

#pragma once

#include <memory>

#include "pmlc/util/enums.h"

namespace mlir {
class Pass;
} // namespace mlir

namespace pmlc::dialect::tile {

class ContractionOp;

std::unique_ptr<mlir::Pass> createComputeBoundsPass();

std::unique_ptr<mlir::Pass> createConstantTypesPass();

std::unique_ptr<mlir::Pass> createConstantTypesPass(mlir::Type concreteFloat,
                                                    mlir::Type concreteInt);

std::unique_ptr<mlir::Pass> createPadPass();

} // namespace pmlc::dialect::tile
