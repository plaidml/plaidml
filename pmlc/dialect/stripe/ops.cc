// Copyright 2019, Intel Corporation
#include "pmlc/dialect/stripe/ops.h"

#include <string>
#include <vector>

#include "mlir/IR/OpImplementation.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Support/DebugStringHelper.h"

#include "pmlc/dialect/stripe/dialect.h"
#include "pmlc/dialect/stripe/rewrites.h"

namespace pmlc::dialect::stripe {

#include "pmlc/dialect/stripe/ops_interfaces.cc.inc"

void AffinePolyOp::getCanonicalizationPatterns(OwningRewritePatternList& results, MLIRContext* context) {
  results.insert<SimplifyPoly>(context);
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
  // If we can have varargs, print them wrapped in ()'s
  if (vararg) {
    if (count == 0) {
      *p << " ";
    }
    *p << "(";
    p->printOperands(op->operand_begin() + count, op->operand_end());
    *p << ")";
  }
  // Print the fixed attributes (which are always integers in our case)
  bool first = (count == 0);
  for (StringRef name : fixed) {
    if (!first) {
      *p << ", ";
    } else {
      *p << " ";
    }
    first = false;
    if (auto at = op->getAttrOfType<IntegerAttr>(name)) {
      *p << at.getValue();
    } else if (auto at = op->getAttrOfType<ArrayAttr>(name)) {
      *p << "[";
      for (size_t i = 0; i < at.getValue().size(); i++) {
        if (auto val = at.getValue()[i].dyn_cast<IntegerAttr>()) {
          if (i != 0) *p << ", ";
          *p << val.getValue();
        } else {
          throw std::runtime_error("Invalid attribute, array isn't integers");
        }
      }
      *p << "]";
    } else {
      op->getAttr(name).dump();
      throw std::runtime_error("Invalid attribute: " + std::string(name));
    }
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
  // If we can have varargs, parse them wrapped in ()'s
  if (vararg) {
    r = r || p->parseOperandList(*ops, -1, OpAsmParser::Delimiter::Paren);
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
  result.addAttribute("layout", TypeAttr::get(type));
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

static void printAffinePolyOp(OpAsmPrinter& p, AffinePolyOp op) {  // NOLINT
  PrintSimple(op.getOperation(), &p, 0, {"coeffs", "offset"}, Type(), true);
}

static ParseResult parseAffinePolyOp(OpAsmParser& parser, OperationState& result) {  // NOLINT
  auto aff_type = AffineType::get(parser.getBuilder().getContext());
  llvm::SmallVector<OpAsmParser::OperandType, 0> operands;
  bool r = ParseSimple(&parser, &result, &operands, 0, {"coeffs", "offset"}, static_cast<Type*>(nullptr), true);
  r = r || parser.resolveOperands(operands, aff_type, result.operands);
  r = r || parser.addTypeToList(aff_type, result.types);
  return mlir::failure(r);
}

void AffinePolyOp::build(Builder* builder, OperationState& result, const AffinePolynomial& poly) {  // NOLINT
  llvm::SmallVector<int64_t, 8> coeffs;
  for (const auto& kvp : poly.terms) {
    result.addOperands(kvp.first);
    coeffs.push_back(kvp.second);
  }
  result.addAttribute("coeffs", builder->getI64ArrayAttr(coeffs));
  result.addAttribute("offset", builder->getI64IntegerAttr(poly.constant));
  result.addTypes(builder->getType<AffineType>());
  result.setOperandListToResizable();
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

void ConstraintOp::getCanonicalizationPatterns(OwningRewritePatternList& results, MLIRContext* context) {
  results.insert<RemoveTrivialConstraints>(context);
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
