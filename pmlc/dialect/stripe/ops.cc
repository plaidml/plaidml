// Copyright 2019, Intel Corporation
#include "pmlc/dialect/stripe/ops.h"

#include <string>
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

static void printTerminateOp(OpAsmPrinter& p, TerminateOp op) {  // NOLINT
  p << op.getOperationName();
}

static ParseResult parseTerminateOp(OpAsmParser& parser, OperationState& result) {  // NOLINT
  return mlir::success();
}

static void printAllocateOp(OpAsmPrinter& p, AllocateOp op) {  // NOLINT
  PrintSimple(op.getOperation(), &p, 0, {}, Type(), false);
}

static ParseResult parseAllocateOp(OpAsmParser& parser, OperationState& result) {  // NOLINT
  llvm::SmallVector<OpAsmParser::OperandType, 0> operands;
  bool r = ParseSimple(&parser, &result, &operands, 0, {}, static_cast<Type*>(nullptr), false);
  for (auto& kvp : result.attributes) {
    if (kvp.first.str() != "layout") continue;
    TypeAttr ta = kvp.second.dyn_cast<TypeAttr>();
    if (!ta) continue;
    auto layout = ta.getValue().dyn_cast<TensorType>();
    if (!layout) continue;
    result.addTypes(TensorRefType::get(layout.getElementType(), layout.getRank(), layout.is_const()));
  }
  return mlir::failure(r);
}

void AllocateOp::build(Builder* builder, OperationState& result, TensorType type) {  // NOLINT
  result.addAttribute("layout", builder->getTypeAttr(type));
  result.addTypes(TensorRefType::get(type));
}

static void printRefineOp(OpAsmPrinter& p, RefineOp op) {  // NOLINT
  PrintSimple(op.getOperation(), &p, 1, {}, op.in()->getType(), true);
}

static ParseResult parseRefineOp(OpAsmParser& parser, OperationState& result) {  // NOLINT
  TensorRefType refType;
  auto aff_type = AffineType::get(parser.getBuilder().getContext());
  llvm::SmallVector<OpAsmParser::OperandType, 8> operands;
  bool r = ParseSimple(&parser, &result, &operands, 1, {}, &refType, true);
  r = r || parser.resolveOperand(operands[0], refType, result.operands);
  for (size_t i = 1; i < operands.size(); i++) {
    r = r || parser.resolveOperand(operands[i], aff_type, result.operands);
  }
  r = r || parser.addTypeToList(refType, result.types);
  return mlir::failure(r);
}

static void printLoadOp(OpAsmPrinter& p, LoadOp op) {  // NOLINT
  PrintSimple(op.getOperation(), &p, 1, {}, op.from()->getType(), false);
}

static ParseResult parseLoadOp(OpAsmParser& parser, OperationState& result) {  // NOLINT
  TensorRefType refType;
  llvm::SmallVector<OpAsmParser::OperandType, 1> operands;
  bool r = ParseSimple(&parser, &result, &operands, 1, {}, &refType, false);
  Type eltType = RankedTensorType::get({}, refType.getElementType());
  r = r || parser.resolveOperand(operands[0], refType, result.operands);
  r = r || parser.addTypeToList(eltType, result.types);
  return mlir::failure(r);
}

static void printStoreOp(OpAsmPrinter& p, StoreOp op) {  // NOLINT
  PrintSimple(op.getOperation(), &p, 2, {}, op.into()->getType(), false);
}

static ParseResult parseStoreOp(OpAsmParser& parser, OperationState& result) {  // NOLINT
  TensorRefType refType;
  llvm::SmallVector<OpAsmParser::OperandType, 2> operands;
  bool r = ParseSimple(&parser, &result, &operands, 2, {}, &refType, false);
  Type eltType = RankedTensorType::get({}, refType.getElementType());
  r = r || parser.resolveOperand(operands[0], refType, result.operands);
  r = r || parser.resolveOperand(operands[1], eltType, result.operands);
  return mlir::failure(r);
}

static void printAggregateOp(OpAsmPrinter& p, AggregateOp op) {  // NOLINT
  p << op.getOperation()->getName() << " \"";
  p << util::stringifyAggregationKind(op.agg());
  p << "\" ";
  p.printOperand(op.into());
  p << " ";
  p.printOperand(op.from());
  p << " : " << op.into()->getType();
  llvm::SmallVector<StringRef, 1> skip;
  skip.push_back("agg");
  p.printOptionalAttrDict(op.getOperation()->getAttrs(), skip);
}

static ParseResult parseAggregateOp(OpAsmParser& parser, OperationState& result) {  // NOLINT
  bool r = false;
  StringAttr agg_op_val;
  llvm::SmallVector<NamedAttribute, 1> ignore;
  MLIRContext* ctx = parser.getBuilder().getContext();
  r = r || parser.parseAttribute(agg_op_val, mlir::NoneType::get(ctx), "agg", ignore);
  auto agg_op = util::symbolizeAggregationKind(agg_op_val.getValue());
  if (!agg_op) {
    return mlir::failure();
  }
  auto agg_op_attr = parser.getBuilder().getI64IntegerAttr(static_cast<int64_t>(agg_op.getValue()));
  result.attributes.emplace_back(mlir::Identifier::get("agg", ctx), agg_op_attr);
  OpAsmParser::OperandType into;
  OpAsmParser::OperandType from;
  r = r || parser.parseOperand(into);
  r = r || parser.parseOperand(from);
  TensorRefType refType;
  r = r || parser.parseColonType(refType);
  Type eltType = RankedTensorType::get({}, refType.getElementType());
  r = r || parser.resolveOperand(into, refType, result.operands);
  r = r || parser.resolveOperand(from, eltType, result.operands);
  return mlir::failure(r);
}

static void printAffineConstOp(OpAsmPrinter& p, AffineConstOp op) {  // NOLINT
  PrintSimple(op.getOperation(), &p, 0, {"value"}, Type(), false);
}

static ParseResult parseAffineConstOp(OpAsmParser& parser, OperationState& result) {  // NOLINT
  auto aff_type = AffineType::get(parser.getBuilder().getContext());
  llvm::SmallVector<OpAsmParser::OperandType, 0> operands;
  bool r = ParseSimple(&parser, &result, &operands, 0, {"value"}, static_cast<Type*>(nullptr), false);
  r = r || parser.addTypeToList(aff_type, result.types);
  return mlir::failure(r);
}

static void printAffineMulOp(OpAsmPrinter& p, AffineMulOp op) {  // NOLINT
  PrintSimple(op.getOperation(), &p, 1, {"scale"}, Type(), false);
}

static ParseResult parseAffineMulOp(OpAsmParser& parser, OperationState& result) {  // NOLINT
  auto aff_type = AffineType::get(parser.getBuilder().getContext());
  llvm::SmallVector<OpAsmParser::OperandType, 1> operands;
  bool r = ParseSimple(&parser, &result, &operands, 1, {"scale"}, static_cast<Type*>(nullptr), false);
  r = r || parser.resolveOperand(operands[0], aff_type, result.operands);
  r = r || parser.addTypeToList(aff_type, result.types);
  return mlir::failure(r);
}

static void printAffineAddOp(OpAsmPrinter& p, AffineAddOp op) {  // NOLINT
  PrintSimple(op.getOperation(), &p, 0, {}, Type(), true);
}

static ParseResult parseAffineAddOp(OpAsmParser& parser, OperationState& result) {  // NOLINT
  auto aff_type = AffineType::get(parser.getBuilder().getContext());
  llvm::SmallVector<OpAsmParser::OperandType, 4> operands;
  bool r = ParseSimple(&parser, &result, &operands, 0, {}, static_cast<Type*>(nullptr), true);
  r = r || parser.resolveOperands(operands, aff_type, result.operands);
  r = r || parser.addTypeToList(aff_type, result.types);
  return mlir::failure(r);
}

static void printParallelForOp(OpAsmPrinter& p, ParallelForOp op) {  // NOLINT
  p << op.getOperation()->getName() << " (";
  llvm::SmallVector<StringRef, 8> skip;
  skip.push_back("ranges");
  skip.push_back("idx_names");
  auto names = op.getAttrOfType<ArrayAttr>(mlir::Identifier::get("idx_names", op.getContext()));
  for (size_t i = 0; i < op.ranges().size(); i++) {
    auto idx_name = StringAttr::get("", op.getContext());
    if (names && names.size() > i) {
      if (auto str_attr = names.getValue()[i].template dyn_cast<StringAttr>()) {
        idx_name = str_attr;
      }
    }
    p << idx_name << ":" << op.ranges().getValue()[i].cast<IntegerAttr>().getInt();
    if (i + 1 != op.ranges().size()) {
      p << ", ";
    }
  }
  p << ")";
  p.printRegion(op.inner());
  p.printOptionalAttrDict(op.getOperation()->getAttrs(), skip);
}

static ParseResult parseParallelForOp(OpAsmParser& parser, OperationState& result) {  // NOLINT
  bool r = false;
  r = r || parser.parseLParen();
  MLIRContext* ctx = parser.getBuilder().getContext();
  std::vector<std::string> idx_name;
  std::vector<int64_t> idx_range;
  llvm::SmallVector<mlir::Attribute, 8> ranges;
  llvm::SmallVector<mlir::Attribute, 8> idx_names;
  while (!r) {
    if (!parser.parseOptionalRParen()) {
      break;
    }
    parser.parseOptionalComma();
    StringAttr idx_name;
    IntegerAttr idx_range;
    llvm::SmallVector<NamedAttribute, 2> ignore;
    r = r || parser.parseAttribute(idx_name, mlir::NoneType::get(ctx), "name", ignore);
    r = r || parser.parseColon();
    r = r || parser.parseAttribute(idx_range, Type(), "range", ignore);
    if (!r) {
      idx_names.push_back(idx_name);
      ranges.push_back(idx_range);
    }
  }
  auto ranges_attr = ArrayAttr::get(ranges, ctx);
  auto idx_names_attr = ArrayAttr::get(idx_names, ctx);
  result.attributes.emplace_back(mlir::Identifier::get("ranges", ctx), ranges_attr);
  result.attributes.emplace_back(mlir::Identifier::get("idx_names", ctx), idx_names_attr);
  result.regions.emplace_back(new Region(nullptr));
  r = r || parser.parseRegion(*result.regions.back(), {}, {}, false);
  r = r || parser.parseOptionalAttributeDict(result.attributes);
  return mlir::failure(r);
}

void ParallelForOp::build(Builder* builder, OperationState& result, ArrayRef<int64_t> ranges) {  // NOLINT
  result.addAttribute("ranges", builder->getI64ArrayAttr(ranges));
  auto region = result.addRegion();
  Block* body = new Block();
  for (size_t i = 0; i < ranges.size(); i++) {
    body->addArgument(AffineType::get(builder->getContext()));
  }
  region->push_back(body);
  OperationState state(result.location, TerminateOp::getOperationName());
  TerminateOp::build(builder, state);
  body->push_back(Operation::create(state));
}

static void printConstraintOp(OpAsmPrinter& p, ConstraintOp op) {  // NOLINT
  p << op.getOperation()->getName() << " ";
  p.printOperand(op.input());
  p.printRegion(op.ge_case());
  if (!op.lt_case().empty()) {
    p.printRegion(op.lt_case());
  }
}

static ParseResult parseConstraintOp(OpAsmParser& parser, OperationState& result) {  // NOLINT
  auto aff_type = AffineType::get(parser.getBuilder().getContext());
  OpAsmParser::OperandType op;
  bool r = false;
  r = r || parser.parseOperand(op);
  r = r || parser.resolveOperand(op, aff_type, result.operands);
  result.regions.emplace_back(new Region(nullptr));
  r = r || parser.parseRegion(*result.regions.back(), {}, {});
  result.regions.emplace_back(new Region(nullptr));
  r = r || parser.parseOptionalRegion(*result.regions.back(), {}, {});
  return mlir::failure(r);
}

static void printExecuteOnOp(OpAsmPrinter& p, ExecuteOnOp& op) {  // NOLINT
  PrintSimple(op.getOperation(), &p, 1, {}, op.from()->getType(), false);
}

static ParseResult parseExecuteOnOp(OpAsmParser& parser, OperationState& result) {  // NOLINT
  TensorRefType refType;
  llvm::SmallVector<OpAsmParser::OperandType, 1> operands;
  bool r = ParseSimple(&parser, &result, &operands, 1, {}, &refType, false);
  r = r || parser.resolveOperand(operands[0], refType, result.operands);
  return mlir::failure(r);
}

#define GET_OP_CLASSES
#include "pmlc/dialect/stripe/ops.cc.inc"

}  // namespace pmlc::dialect::stripe
