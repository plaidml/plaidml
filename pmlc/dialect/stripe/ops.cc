// Copyright 2019, Intel Corporation
#include "pmlc/dialect/stripe/ops.h"

#include <string>
#include <vector>

#include "mlir/IR/OpImplementation.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Support/DebugStringHelper.h"

#include "pmlc/dialect/eltwise/util.h"
#include "pmlc/dialect/stripe/dialect.h"
#include "pmlc/dialect/stripe/rewrites.h"

namespace pmlc::dialect::stripe {

using mlir::failure;
using mlir::success;

void AffinePolyOp::getCanonicalizationPatterns(OwningRewritePatternList& results, MLIRContext* context) {
  results.insert<SimplifyPoly>(context);
}

void RefineOp::getCanonicalizationPatterns(OwningRewritePatternList& results, MLIRContext* context) {
  results.insert<SimplifyNopRefines>(context);
}

void ConstraintOp::getCanonicalizationPatterns(OwningRewritePatternList& results, MLIRContext* context) {
  results.insert<RemoveTrivialConstraints>(context);
}

void ParallelForOp::getCanonicalizationPatterns(OwningRewritePatternList& results, MLIRContext* context) {
  results.insert<InlineNoIndexParallelFors>(context);
  results.insert<RemoveRangeZeroParallelFors>(context);
  results.insert<RemoveNoSideEffectParallelFors>(context);
  results.insert<RemoveRangeOneIndexes>(context);
}

void PrintSimple(               //
    Operation* op,              //
    OpAsmPrinter* printer,      //
    size_t count,               //
    ArrayRef<StringRef> fixed,  //
    Type otype,                 //
    bool vararg) {
  // Print the op name
  *printer << op->getName();
  if (count > 0) {
    *printer << " ";
  }
  // Print the normal (fixed) operands
  printer->printOperands(op->operand_begin(), op->operand_begin() + count);
  // If we can have varargs, print them wrapped in ()'s
  if (vararg) {
    if (count == 0) {
      *printer << " ";
    }
    *printer << "(";
    printer->printOperands(op->operand_begin() + count, op->operand_end());
    *printer << ")";
  }
  // Print the fixed attributes (which are always integers in our case)
  bool first = (count == 0);
  for (StringRef name : fixed) {
    if (!first) {
      *printer << ", ";
    } else {
      *printer << " ";
    }
    first = false;
    if (auto at = op->getAttrOfType<IntegerAttr>(name)) {
      *printer << at.getValue();
    } else if (auto at = op->getAttrOfType<ArrayAttr>(name)) {
      *printer << "[";
      for (size_t i = 0; i < at.getValue().size(); i++) {
        if (auto val = at.getValue()[i].dyn_cast<IntegerAttr>()) {
          if (i != 0) *printer << ", ";
          *printer << val.getValue();
        } else {
          throw std::runtime_error("Invalid attribute, array isn't integers");
        }
      }
      *printer << "]";
    } else {
      op->getAttr(name).dump();
      throw std::runtime_error("Invalid attribute: " + std::string(name));
    }
  }
  // Print a type (if needed)
  if (otype) {
    *printer << " : " << otype;
  }
  // Print any additional attributes
  printer->printOptionalAttrDict(op->getAttrs(), fixed);
}

template <typename T>
ParseResult ParseSimple(                                   //
    OpAsmParser* parser,                                   //
    OperationState* res,                                   //
    llvm::SmallVectorImpl<OpAsmParser::OperandType>* ops,  //
    size_t count,                                          //
    ArrayRef<StringRef> fixed,                             //
    T* out_type,                                           //
    bool vararg) {
  // Parse the normal operands, annoyingly parseOperandList doesn't
  // have an option to read exactly N operands, only to read all and verify
  for (size_t i = 0; i < count; i++) {
    if (i && parser->parseComma()) {
      return failure();
    }
    OpAsmParser::OperandType op;
    if (parser->parseOperand(op)) {
      return failure();
    }
    ops->push_back(op);
  }
  // If we can have varargs, parse them wrapped in ()'s
  if (vararg && parser->parseOperandList(*ops, -1, OpAsmParser::Delimiter::Paren)) {
    return failure();
  }
  // Parse the fixed attributes
  for (unsigned i = 0; i < fixed.size(); i++) {
    if (i && parser->parseComma()) {
      return failure();
    }
    Attribute ignore;
    if (parser->parseAttribute(ignore, fixed[i], res->attributes)) {
      return failure();
    }
  }
  // Parse a type if needed
  if (out_type) {
    if (parser->parseColon() || parser->parseType(*out_type)) {
      return failure();
    }
  }
  // Parse any additional attributes
  return parser->parseOptionalAttributeDict(res->attributes);
}

void printTerminateOp(OpAsmPrinter* printer, TerminateOp op) {  //
  *printer << op.getOperationName();
}

ParseResult parseTerminateOp(OpAsmParser* parser, OperationState& result) {  //
  return mlir::success();
}

void printAllocateOp(OpAsmPrinter* printer, AllocateOp op) {
  PrintSimple(op.getOperation(), printer, 0, {}, Type(), false);
}

ParseResult parseAllocateOp(OpAsmParser* parser, OperationState& result) {
  llvm::SmallVector<OpAsmParser::OperandType, 0> operands;
  if (ParseSimple(parser, &result, &operands, 0, {}, static_cast<Type*>(nullptr), false)) {
    return failure();
  }
  for (auto& kvp : result.attributes) {
    if (kvp.first.str() != "layout") {
      continue;
    }
    TypeAttr ta = kvp.second.dyn_cast<TypeAttr>();
    if (!ta) {
      continue;
    }
    auto layout = ta.getValue().dyn_cast<TensorType>();
    if (!layout) {
      continue;
    }
    result.addTypes(TensorRefType::get(layout.getElementType(), layout.getRank(), layout.is_const()));
  }
  return success();
}

void AllocateOp::build(Builder* builder, OperationState& result, TensorType type) {
  result.addAttribute("layout", TypeAttr::get(type));
  result.addTypes(TensorRefType::get(type));
}

void printRefineOp(OpAsmPrinter* printer, RefineOp op) {
  PrintSimple(op.getOperation(), printer, 1, {}, op.in()->getType(), true);
}

ParseResult parseRefineOp(OpAsmParser* parser, OperationState& result) {
  TensorRefType refType;
  llvm::SmallVector<OpAsmParser::OperandType, 8> operands;
  if (ParseSimple(parser, &result, &operands, 1, {}, &refType, true) ||  //
      parser->resolveOperand(operands[0], refType, result.operands)) {
    return failure();
  }
  auto affineType = AffineType::get(parser->getBuilder().getContext());
  for (size_t i = 1; i < operands.size(); i++) {
    if (parser->resolveOperand(operands[i], affineType, result.operands)) {
      return failure();
    }
  }
  return parser->addTypeToList(refType, result.types);
}

void printLoadOp(OpAsmPrinter* printer, LoadOp op) {
  PrintSimple(op.getOperation(), printer, 1, {}, op.from()->getType(), false);
}

ParseResult parseLoadOp(OpAsmParser* parser, OperationState& result) {
  TensorRefType refType;
  llvm::SmallVector<OpAsmParser::OperandType, 1> operands;
  return failure(  //
      ParseSimple(parser, &result, &operands, 1, {}, &refType, false) ||
      parser->resolveOperand(operands[0], refType, result.operands) ||
      parser->addTypeToList(eltwise::getRankedTensorType(refType.getElementType()), result.types));
}

void printStoreOp(OpAsmPrinter* printer, StoreOp op) {
  PrintSimple(op.getOperation(), printer, 2, {}, op.into()->getType(), false);
}

ParseResult parseStoreOp(OpAsmParser* parser, OperationState& result) {
  TensorRefType refType;
  llvm::SmallVector<OpAsmParser::OperandType, 2> operands;
  return failure(  //
      ParseSimple(parser, &result, &operands, 2, {}, &refType, false) ||
      parser->resolveOperand(operands[0], refType, result.operands) ||
      parser->resolveOperand(operands[1], eltwise::getRankedTensorType(refType.getElementType()), result.operands));
}

void printAggregateOp(OpAsmPrinter* printer, AggregateOp op) {
  *printer << op.getOperation()->getName() << " \"";
  *printer << util::stringifyAggregationKind(op.agg());
  *printer << "\" ";
  printer->printOperand(op.into());
  *printer << " ";
  printer->printOperand(op.from());
  *printer << " : " << op.into()->getType();
  llvm::SmallVector<StringRef, 1> skip;
  skip.push_back("agg");
  printer->printOptionalAttrDict(op.getOperation()->getAttrs(), skip);
}

ParseResult parseAggregateOp(OpAsmParser* parser, OperationState& result) {
  StringAttr agg_op_val;
  llvm::SmallVector<NamedAttribute, 1> ignore;
  auto ctx = parser->getBuilder().getContext();
  if (parser->parseAttribute(agg_op_val, mlir::NoneType::get(ctx), "agg", ignore)) {
    return failure();
  }
  auto agg_op = util::symbolizeAggregationKind(agg_op_val.getValue());
  if (!agg_op) {
    return failure();
  }
  auto agg_op_attr = parser->getBuilder().getI64IntegerAttr(static_cast<int64_t>(agg_op.getValue()));
  result.attributes.emplace_back(mlir::Identifier::get("agg", ctx), agg_op_attr);
  OpAsmParser::OperandType into;
  OpAsmParser::OperandType from;
  TensorRefType refType;
  return failure(                         //
      parser->parseOperand(into) ||       //
      parser->parseOperand(from) ||       //
      parser->parseColonType(refType) ||  //
      parser->resolveOperand(into, refType, result.operands) ||
      parser->resolveOperand(from, eltwise::getRankedTensorType(refType.getElementType()), result.operands));
}

void printAffinePolyOp(OpAsmPrinter* printer, AffinePolyOp op) {
  PrintSimple(op.getOperation(), printer, 0, {"coeffs", "offset"}, Type(), true);
}

ParseResult parseAffinePolyOp(OpAsmParser* parser, OperationState& result) {
  auto aff_type = AffineType::get(parser->getBuilder().getContext());
  llvm::SmallVector<OpAsmParser::OperandType, 0> operands;
  return failure(  //
      ParseSimple(parser, &result, &operands, 0, {"coeffs", "offset"}, static_cast<Type*>(nullptr), true) ||
      parser->resolveOperands(operands, aff_type, result.operands) ||  //
      parser->addTypeToList(aff_type, result.types));
}

void AffinePolyOp::build(Builder* builder, OperationState& result, const AffinePolynomial& poly) {
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

void printParallelForOp(OpAsmPrinter* printer, ParallelForOp op) {
  *printer << op.getOperation()->getName() << " (";
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
    *printer << idx_name << ":" << op.ranges().getValue()[i].cast<IntegerAttr>().getInt();
    if (i + 1 != op.ranges().size()) {
      *printer << ", ";
    }
  }
  *printer << ")";
  printer->printRegion(op.inner());
  printer->printOptionalAttrDict(op.getOperation()->getAttrs(), skip);
}

ParseResult parseParallelForOp(OpAsmParser* parser, OperationState& result) {
  if (parser->parseLParen()) {
    return failure();
  }
  auto ctx = parser->getBuilder().getContext();
  llvm::SmallVector<mlir::Attribute, 8> ranges;
  llvm::SmallVector<mlir::Attribute, 8> idx_names;
  bool failed = false;
  while (!failed) {
    if (!parser->parseOptionalRParen()) {
      break;
    }
    parser->parseOptionalComma();
    StringAttr idx_name;
    IntegerAttr idx_range;
    llvm::SmallVector<NamedAttribute, 2> ignore;
    failed = (                                                                         //
        parser->parseAttribute(idx_name, mlir::NoneType::get(ctx), "name", ignore) ||  //
        parser->parseColon() ||                                                        //
        parser->parseAttribute(idx_range, Type(), "range", ignore));
    if (!failed) {
      idx_names.push_back(idx_name);
      ranges.push_back(idx_range);
    }
  }
  auto ranges_attr = ArrayAttr::get(ranges, ctx);
  auto idx_names_attr = ArrayAttr::get(idx_names, ctx);
  result.attributes.emplace_back(mlir::Identifier::get("ranges", ctx), ranges_attr);
  result.attributes.emplace_back(mlir::Identifier::get("idx_names", ctx), idx_names_attr);
  result.regions.emplace_back(new Region(nullptr));
  return failure(                                                    //
      parser->parseRegion(*result.regions.back(), {}, {}, false) ||  //
      parser->parseOptionalAttributeDict(result.attributes));
}

void ParallelForOp::build(Builder* builder, OperationState& result, ArrayRef<int64_t> ranges) {
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

void printConstraintOp(OpAsmPrinter* printer, ConstraintOp op) {
  *printer << op.getOperation()->getName() << " ";
  printer->printOperand(op.input());
  printer->printRegion(op.ge_case());
  if (!op.lt_case().empty()) {
    printer->printRegion(op.lt_case());
  }
}

ParseResult parseConstraintOp(OpAsmParser* parser, OperationState& result) {
  auto aff_type = AffineType::get(parser->getBuilder().getContext());
  auto geRegion = new Region(nullptr);
  result.regions.emplace_back(geRegion);
  auto ltRegion = new Region(nullptr);
  result.regions.emplace_back(ltRegion);
  OpAsmParser::OperandType op;
  return failure(                                               //
      parser->parseOperand(op) ||                               //
      parser->resolveOperand(op, aff_type, result.operands) ||  //
      parser->parseRegion(*geRegion, {}, {}) ||                 //
      parser->parseOptionalRegion(*ltRegion, {}, {}));
}

void printExecuteOnOp(OpAsmPrinter* printer, ExecuteOnOp& op) {
  PrintSimple(op.getOperation(), printer, 1, {}, op.from()->getType(), false);
}

ParseResult parseExecuteOnOp(OpAsmParser* parser, OperationState& result) {
  TensorRefType refType;
  llvm::SmallVector<OpAsmParser::OperandType, 1> operands;
  return failure(  //
      ParseSimple(parser, &result, &operands, 1, {}, &refType, false) ||
      parser->resolveOperand(operands[0], refType, result.operands));
}

#include "pmlc/dialect/stripe/ops_interfaces.cc.inc"

#define GET_OP_CLASSES
#include "pmlc/dialect/stripe/ops.cc.inc"

}  // namespace pmlc::dialect::stripe
