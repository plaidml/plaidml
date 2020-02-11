// Copyright 2020 Intel Corporation

#pragma once

#include <memory>

namespace mlir {
template <typename T>
class OpPassBase;
}  // namespace mlir

namespace pmlc::dialect::tile {

class ContractionOp;

std::unique_ptr<mlir::OpPassBase<ContractionOp>> createComputeBoundsPass();

}  // namespace pmlc::dialect::tile
