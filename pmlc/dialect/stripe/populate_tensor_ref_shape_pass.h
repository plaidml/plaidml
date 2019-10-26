// Copyright 2019 Intel Corporation

#pragma once

#include <memory>

namespace mlir {
class FuncOp;
class MLIRContext;
template <typename T>
class OpPassBase;
using FunctionPassBase = OpPassBase<FuncOp>;
class OwningRewritePatternList;

/// Creates a pass to populate 'tensor_ref' types with shape information.
std::unique_ptr<FunctionPassBase> createPopulateTensorRefShapePass();
}  // namespace mlir
