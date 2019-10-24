// Copyright 2019, Intel Corporation
#include "pmlc/dialect/stripe/ops.h"

#include <vector>

#include "mlir/IR/OpImplementation.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Support/DebugStringHelper.h"

namespace pmlc::dialect::stripe {

#include "pmlc/dialect/stripe/ops_interfaces.cc.inc"

namespace {

struct SimplifyAddNothing final : public mlir::OpRewritePattern<AffineAddOp> {
  explicit SimplifyAddNothing(mlir::MLIRContext* context) : OpRewritePattern<AffineAddOp>(context) {}

  mlir::PatternMatchResult match(AffineAddOp op) const final {
    if (!op.getNumOperands()) {
      return matchSuccess();
    }
    return matchFailure();
  }

  void rewrite(AffineAddOp op, mlir::PatternRewriter& rewriter) const final {  // NOLINT(runtime/references)
    rewriter.replaceOpWithNewOp<AffineConstOp>(op, rewriter.getType<AffineType>(), rewriter.getI64IntegerAttr(0));
  }
};

struct SimplifyAddSingle final : public mlir::OpRewritePattern<AffineAddOp> {
  explicit SimplifyAddSingle(mlir::MLIRContext* context) : OpRewritePattern<AffineAddOp>(context) {}

  mlir::PatternMatchResult match(AffineAddOp op) const final {
    if (op.getNumOperands() == 1) {
      return matchSuccess();
    }
    return matchFailure();
  }

  void rewrite(AffineAddOp op, mlir::PatternRewriter& rewriter) const final {  // NOLINT(runtime/references)
    rewriter.replaceOp(op, op.getOperand(0));
  }
};

struct SimplifyAddConstants final : public mlir::OpRewritePattern<AffineAddOp> {
  explicit SimplifyAddConstants(mlir::MLIRContext* context) : OpRewritePattern<AffineAddOp>(context) {}

  mlir::PatternMatchResult matchAndRewrite(AffineAddOp op,
                                           mlir::PatternRewriter& rewriter) const final {  // NOLINT(runtime/references)
    std::size_t constCount = 0;
    std::int64_t constSum = 0;

    for (auto* operand : op.inputs()) {
      auto val = mlir::dyn_cast_or_null<AffineConstOp>(operand->getDefiningOp());
      if (val) {
        ++constCount;
        constSum += val.value().getSExtValue();
      }
    }

    if (constCount < 2) {
      return matchFailure();
    }

    std::vector<mlir::Value*> operands;
    operands.reserve(op.getNumOperands() - constCount + (constSum ? 1 : 0));
    for (auto* operand : op.inputs()) {
      auto val = mlir::dyn_cast_or_null<AffineConstOp>(operand->getDefiningOp());
      if (!val) {
        operands.emplace_back(operand);
      }
    }
    if (constSum) {
      operands.emplace_back(rewriter.create<AffineConstOp>(op.getLoc(), rewriter.getType<AffineType>(),
                                                           rewriter.getI64IntegerAttr(constSum)));
    }
    rewriter.replaceOpWithNewOp<AffineAddOp>(op, rewriter.getType<AffineType>(), operands);

    return matchSuccess();
  }
};

struct SimplifyMulConstOne final : public mlir::OpRewritePattern<AffineMulOp> {
  explicit SimplifyMulConstOne(mlir::MLIRContext* context) : OpRewritePattern<AffineMulOp>(context) {}

  mlir::PatternMatchResult match(AffineMulOp op) const final {
    if (op.scale().getSExtValue() == 1) {
      return matchSuccess();
    }
    return matchFailure();
  }

  void rewrite(AffineMulOp op, mlir::PatternRewriter& rewriter) const final {  // NOLINT(runtime/references)
    rewriter.replaceOp(op, op.input());
  }
};

struct SimplifyMulConstValue final : public mlir::OpRewritePattern<AffineMulOp> {
  explicit SimplifyMulConstValue(mlir::MLIRContext* context) : OpRewritePattern<AffineMulOp>(context) {}

  mlir::PatternMatchResult match(AffineMulOp op) const final {
    auto val = mlir::dyn_cast_or_null<AffineConstOp>(op.input()->getDefiningOp());
    if (!val) {
      return matchFailure();
    }
    return matchSuccess();
  }

  void rewrite(AffineMulOp op, mlir::PatternRewriter& rewriter) const final {  // NOLINT(runtime/references)
    auto val = mlir::cast<AffineConstOp>(op.input()->getDefiningOp());
    auto constValue = val.value().getSExtValue();
    if (constValue == 0) {
      rewriter.replaceOp(op, op.input());  // A handy source of a zero
    } else {
      auto maybeDead = llvm::SmallVector<mlir::Value*, 1>{op.input()};
      rewriter.replaceOpWithNewOp<AffineConstOp>(maybeDead, op, rewriter.getType<AffineType>(),
                                                 rewriter.getI64IntegerAttr((op.scale() * constValue).getSExtValue()));
    }
  }
};

struct SimplifyMulConstZero final : public mlir::OpRewritePattern<AffineMulOp> {
  explicit SimplifyMulConstZero(mlir::MLIRContext* context) : OpRewritePattern<AffineMulOp>(context) {}

  mlir::PatternMatchResult match(AffineMulOp op) const final {
    if (!op.scale()) {
      return matchSuccess();
    }
    return matchFailure();
  }

  void rewrite(AffineMulOp op, mlir::PatternRewriter& rewriter) const final {  // NOLINT(runtime/references)
    auto maybeDead = llvm::SmallVector<mlir::Value*, 1>{op.input()};
    rewriter.replaceOpWithNewOp<AffineConstOp>(maybeDead, op, rewriter.getType<AffineType>(),
                                               rewriter.getI64IntegerAttr(0));
  }
};

struct SimplifyNopRefines final : public mlir::OpRewritePattern<RefineOp> {
  explicit SimplifyNopRefines(mlir::MLIRContext* context) : OpRewritePattern<RefineOp>(context) {}

  mlir::PatternMatchResult match(RefineOp op) const final {
    for (auto* offset : op.offsets()) {
      auto val = mlir::dyn_cast_or_null<AffineConstOp>(offset->getDefiningOp());
      if (!val || !!val.value()) {
        return matchFailure();
      }
    }
    return matchSuccess();
  }

  void rewrite(RefineOp op, mlir::PatternRewriter& rewriter) const final {  // NOLINT(runtime/references)
    rewriter.replaceOp(op, op.in());
  }
};

}  // namespace

void AffineAddOp::getCanonicalizationPatterns(OwningRewritePatternList& results, MLIRContext* context) {
  results.insert<SimplifyAddNothing>(context);
  results.insert<SimplifyAddSingle>(context);
  results.insert<SimplifyAddConstants>(context);
}

void AffineMulOp::getCanonicalizationPatterns(OwningRewritePatternList& results, MLIRContext* context) {
  results.insert<SimplifyMulConstOne>(context);
  results.insert<SimplifyMulConstValue>(context);
  results.insert<SimplifyMulConstZero>(context);
}

void RefineOp::getCanonicalizationPatterns(OwningRewritePatternList& results, MLIRContext* context) {
  results.insert<SimplifyNopRefines>(context);
}

void PrintSimple(Operation* op, OpAsmPrinter* p, size_t count, ArrayRef<StringRef> fixed, Type otype, bool vararg) {
  // Print the op name
  *p << op->getName();
  if (count > 0) {
    *p << " ";
  }
  // Pring the normal (fixed) operands
  p->printOperands(op->operand_begin(), op->operand_begin() + count);
  // Print the fixed attributes (which are always integers in our case)
  bool first = (count == 0);
  for (StringRef name : fixed) {
    if (!first) {
      *p << ", ";
    } else {
      *p << " ";
    }
    first = false;
    *p << op->getAttrOfType<IntegerAttr>(name).getValue();
  }
  // If we can have varargs, print them wrapped in ()'s
  if (vararg) {
    if (count > 0) {
      *p << " ";
    }
    *p << "(";
    p->printOperands(op->operand_begin() + count, op->operand_end());
    *p << ")";
  }
  // Print a type (if needed)
  if (otype) {
    *p << " : " << otype;
  }
  // Print any additional attributes
  p->printOptionalAttrDict(op->getAttrs(), fixed);
}

template <typename T>
bool ParseSimple(OpAsmParser* p, OperationState* res, llvm::SmallVectorImpl<OpAsmParser::OperandType>* ops,
                 size_t count, ArrayRef<StringRef> fixed, T* out_type, bool vararg) {
  bool r = false;
  // Parse the normal operands, annoyingly parseOperandList doesn't
  // have an option to read exactly N operands, only to read all and verify
  bool first = true;
  for (size_t i = 0; i < count; i++) {
    if (!first) {
      r = r || p->parseComma();
    }
    first = false;
    OpAsmParser::OperandType op;
    r = r || p->parseOperand(op);
    ops->push_back(op);
  }
  // Parse the fixed attributes
  for (StringRef name : fixed) {
    if (!first) {
      r = r || p->parseComma();
    }
    Attribute dont_care;
    r = r || p->parseAttribute(dont_care, name, res->attributes);
    first = false;
  }
  // If we can have varargs, parse them wrapped in ()'s
  if (vararg) {
    r = r || p->parseOperandList(*ops, -1, OpAsmParser::Delimiter::Paren);
  }
  // Parse a type if needed
  if (out_type) {
    r = r || p->parseColon();
    r = r || p->parseType(*out_type);
  }
  // Parse any additional attributes
  r = r || p->parseOptionalAttributeDict(res->attributes);
  return r;
}

#define GET_OP_CLASSES
#include "pmlc/dialect/stripe/ops.cc.inc"

}  // namespace pmlc::dialect::stripe
