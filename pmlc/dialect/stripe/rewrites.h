// Copyright 2019, Intel Corporation

#pragma once

#include "mlir/IR/PatternMatch.h"
#include "pmlc/dialect/stripe/ops.h"

namespace pmlc::dialect::stripe {

struct SimplifyPoly final : public mlir::OpRewritePattern<AffinePolyOp> {
  explicit SimplifyPoly(mlir::MLIRContext* context, mlir::PatternBenefit benefit = 1)
      : OpRewritePattern<AffinePolyOp>(context, benefit) {}
  mlir::PatternMatchResult matchAndRewrite(AffinePolyOp op, mlir::PatternRewriter& rewriter) const override;
};

struct SimplifyNopRefines final : public mlir::OpRewritePattern<RefineOp> {
  explicit SimplifyNopRefines(mlir::MLIRContext* context, mlir::PatternBenefit benefit = 1)
      : OpRewritePattern<RefineOp>(context, benefit) {}
  mlir::PatternMatchResult match(RefineOp op) const final;
  void rewrite(RefineOp op, mlir::PatternRewriter& rewriter) const final;
};

struct InlineNoIndexParallelFors final : public mlir::OpRewritePattern<ParallelForOp> {
  explicit InlineNoIndexParallelFors(mlir::MLIRContext* context, mlir::PatternBenefit benefit = 1)
      : OpRewritePattern<ParallelForOp>(context, benefit) {}
  mlir::PatternMatchResult match(ParallelForOp op) const final;
  void rewrite(ParallelForOp op, mlir::PatternRewriter& rewriter) const final;
};

struct RemoveRangeZeroParallelFors final : public mlir::OpRewritePattern<ParallelForOp> {
  explicit RemoveRangeZeroParallelFors(mlir::MLIRContext* context, mlir::PatternBenefit benefit = 1)
      : OpRewritePattern<ParallelForOp>(context, benefit) {}
  mlir::PatternMatchResult match(ParallelForOp op) const final;
  void rewrite(ParallelForOp op, mlir::PatternRewriter& rewriter) const final {  //
    rewriter.replaceOp(op, llvm::None);
  }
};

struct RemoveNoSideEffectParallelFors final : public mlir::OpRewritePattern<ParallelForOp> {
  explicit RemoveNoSideEffectParallelFors(mlir::MLIRContext* context, mlir::PatternBenefit benefit = 1)
      : OpRewritePattern<ParallelForOp>(context, benefit) {}
  mlir::PatternMatchResult match(ParallelForOp op) const final;
  void rewrite(ParallelForOp op, mlir::PatternRewriter& rewriter) const final {  //
    rewriter.replaceOp(op, llvm::None);
  }
};

struct RemoveRangeOneIndexes final : public mlir::OpRewritePattern<ParallelForOp> {
  explicit RemoveRangeOneIndexes(mlir::MLIRContext* context, mlir::PatternBenefit benefit = 1)
      : OpRewritePattern<ParallelForOp>(context, benefit) {}
  mlir::PatternMatchResult match(ParallelForOp op) const final;
  void rewrite(ParallelForOp op, mlir::PatternRewriter& rewriter) const final;
};

struct RemoveTrivialConstraints final : public mlir::OpRewritePattern<ConstraintOp> {
  explicit RemoveTrivialConstraints(mlir::MLIRContext* context, mlir::PatternBenefit benefit = 1)
      : OpRewritePattern<ConstraintOp>(context, benefit) {}
  mlir::PatternMatchResult matchAndRewrite(ConstraintOp op, mlir::PatternRewriter& rewriter) const override;
};

struct SplitParallelFor final : public mlir::OpRewritePattern<ParallelForOp> {
  explicit SplitParallelFor(mlir::MLIRContext* context, mlir::PatternBenefit benefit = 1)
      : OpRewritePattern<ParallelForOp>(context, benefit) {}
  mlir::PatternMatchResult matchAndRewrite(ParallelForOp op, mlir::PatternRewriter& rewriter) const final;
};

}  // namespace pmlc::dialect::stripe
