// Copyright 2020 Intel Corporation

#pragma once

#include <memory>

namespace mlir {
class FuncOp;
template <typename T>
class OpPassBase;
}  // namespace mlir

namespace pmlc::dialect::tile {

class ContractionOp;

std::unique_ptr<mlir::OpPassBase<mlir::FuncOp>> createComputeBoundsPass();

}  // namespace pmlc::dialect::tile
