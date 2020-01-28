// Copyright 2019, Intel Corporation

#pragma once

#include "mlir/Pass/Pass.h"

#include "mlir/IR/Matchers.h"
#include "mlir/IR/PatternMatch.h"

#include "pmlc/dialect/eltwise/ir/ops.h"
#include "pmlc/dialect/tile/ir/ops.h"
#include "pmlc/util/enums.h"

namespace pmlc::dialect::tile {

using mlir::OperationPass;
using mlir::OpRewritePattern;
using mlir::OwningRewritePatternList;
using mlir::PatternMatchResult;
using mlir::PatternRewriter;
using pmlc::dialect::eltwise::ScalarConstantOp;

}  // namespace pmlc::dialect::tile
