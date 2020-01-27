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

struct ConstantTypesRewriter : public OpRewritePattern<ScalarConstantOp> {
  ConstantTypesRewriter(mlir::MLIRContext* context, DataType floatx, DataType intx)
      : OpRewritePattern<ScalarConstantOp>(context), floatx_(floatx), intx_(intx) {}

  DataType floatx_;
  DataType intx_;

  PatternMatchResult matchAndRewrite(ScalarConstantOp constOp, PatternRewriter& rewriter) const override;
};

struct ConstantTypesPass : public OperationPass<ConstantTypesPass> {
  ConstantTypesPass(DataType floatx, DataType intx) : floatx_(floatx), intx_(intx){};

  DataType floatx_;
  DataType intx_;

  void runOnOperation() override {
    OwningRewritePatternList patterns;

    patterns.insert<ConstantTypesRewriter>(&getContext(), floatx_, intx_);

    // TODO: Instead of adding all known patterns from the whole system lazily
    // add and cache the canonicalization patterns for ops we see in practice
    // when building the worklist.  For now, we just grab everything.
    // auto* context = &getContext();
    // for (auto* op : context->getRegisteredOperations()) op->getCanonicalizationPatterns//(patterns, context);

    Operation* op = getOperation();
    applyPatternsGreedily(op->getRegions(), patterns);
  }
};

}  // namespace pmlc::dialect::tile
