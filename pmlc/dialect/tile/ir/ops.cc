// Copyright 2019, Intel Corporation

#include "pmlc/dialect/tile/ir/ops.h"

#include <vector>

#include "mlir/Dialect/StandardOps/IR/Ops.h"
#include "mlir/IR/BlockAndValueMapping.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/Matchers.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Support/DebugStringHelper.h"
#include "mlir/Support/MathExtras.h"

#include "pmlc/dialect/tile/ir/util.h"
#include "pmlc/util/logging.h"
#include "pmlc/util/util.h"

using namespace mlir; // NOLINT

namespace pmlc::dialect::tile {

using llvm::SmallVector;

LogicalResult ArgSortOp::materializeOperands(OpBuilder &builder) {
  return tile::materializeOperands(builder, getOperation());
}

LogicalResult ContractionOp::materializeOperands(OpBuilder &builder) {
  Operation *op = getOperation();
  if (combo() == CombinationKind::cond) {
    auto operands = op->getOpOperands();
    return success(
        succeeded(tile::materializeOperands(
            builder, op,
            llvm::ArrayRef<OpOperand *>{&operands[0], &operands[3]})) &&
        succeeded(tile::materializeOperands(
            builder, op,
            llvm::ArrayRef<OpOperand *>{&operands[1], &operands[2]})));
  }
  return tile::materializeOperands(builder, getOperation());
}

LogicalResult GatherOp::materializeOperands(OpBuilder &builder) {
  Operation *op = getOperation();
  return tile::materializeOperands(builder, op,
                                   op->getOpOperands().take_front());
}

LogicalResult ReshapeOp::materializeOperands(OpBuilder &builder) {
  return tile::materializeOperands(builder, getOperation());
}

LogicalResult ScatterOp::materializeOperands(OpBuilder &builder) {
  Operation *op = getOperation();
  return tile::materializeOperands(builder, op,
                                   op->getOpOperands().take_front());
}

LogicalResult ShapeOp::materializeOperands(OpBuilder &builder) {
  return tile::materializeOperands(builder, getOperation());
}

LogicalResult PragmaOp::materializeOperands(OpBuilder &builder) {
  return tile::materializeOperands(builder, getOperation());
}

// ---- ReshapeOp ----

OpFoldResult ReshapeOp::fold(ArrayRef<Attribute> operands) {
  // reshape(x, x.shape) -> x
  if (tensor().getType() == getType()) {
    return tensor();
  }
  return {};
}

// ---- ContractionOp ----

unsigned ContractionOp::getNumTensors(CombinationKind combo) {
  switch (combo) {
  case CombinationKind::none:
    return 1;
  case CombinationKind::add:
  case CombinationKind::eq:
  case CombinationKind::mul:
    return 2;
  case CombinationKind::cond:
    return 3;
  default:
    throw std::runtime_error("Invalid combination op");
  }
}

void ContractionOp::build(OpBuilder &builder, OperationState &result,
                          Type resultType, Value init, ArrayRef<Value> tensors,
                          AggregationKind agg, CombinationKind combo,
                          AffineMap sink, ArrayRef<AffineMap> srcs,
                          IntegerSet cons, StringRef name) {
  result.addOperands(init);
  result.addOperands(tensors);
  result.addTypes(resultType);
  result.addAttribute("agg",
                      builder.getI64IntegerAttr(static_cast<int64_t>(agg)));
  result.addAttribute("combo",
                      builder.getI64IntegerAttr(static_cast<int64_t>(combo)));
  result.addAttribute(getSinkAttrName(), AffineMapAttr::get(sink));
  result.addAttribute(getSourcesAttrName(),
                      builder.getAffineMapArrayAttr(srcs));
  if (!cons.isEmptyIntegerSet()) {
    result.addAttribute(getConstraintsAttrName(), IntegerSetAttr::get(cons));
  }
  if (name.size()) {
    result.addAttribute("name", builder.getStringAttr(name));
  }
}

AffineMap ContractionOp::getSourceMap(unsigned i) {
  return srcs().getValue()[i].cast<AffineMapAttr>().getValue();
}

void ContractionOp::setLowerBounds(ArrayRef<int64_t> bounds) {
  SmallVector<AffineExpr, 6> exprs;
  for (auto dim : bounds) {
    exprs.push_back(getAffineConstantExpr(dim, getContext()));
  }
  auto map =
      AffineMap::get(/*dimCount=*/0, /*symbolCount=*/0, exprs, getContext());
  (*this)->setAttr(getLowerBoundsAttrName(), AffineMapAttr::get(map));
}

void ContractionOp::setUpperBounds(ArrayRef<int64_t> bounds) {
  SmallVector<AffineExpr, 6> exprs;
  for (auto dim : bounds) {
    exprs.push_back(getAffineConstantExpr(dim, getContext()));
  }
  auto map =
      AffineMap::get(/*dimCount=*/0, /*symbolCount=*/0, exprs, getContext());
  (*this)->setAttr(getUpperBoundsAttrName(), AffineMapAttr::get(map));
}

void ContractionOp::setSink(AffineMap sink) {
  (*this)->setAttr(getSinkAttrName(), AffineMapAttr::get(sink));
}

void ContractionOp::setSources(ArrayRef<AffineMap> srcs) {
  SmallVector<Attribute, 4> attrs;
  for (auto src : srcs) {
    attrs.push_back(AffineMapAttr::get(src));
  }
  (*this)->setAttr(getSourcesAttrName(), ArrayAttr::get(attrs, getContext()));
}

void ContractionOp::setConstraints(IntegerSet cons) {
  if (cons.isEmptyIntegerSet()) {
    removeAttr(getConstraintsAttrName());
  } else {
    (*this)->setAttr(getConstraintsAttrName(), IntegerSetAttr::get(cons));
  }
}

unsigned ContractionOp::getNumTensors() { return getNumTensors(combo()); }

unsigned ContractionOp::getNumSymbols() {
  return getNumOperands() - 1 - getNumTensors();
}

Value ContractionOp::getTensor(unsigned i) {
  return *std::next(operands().begin(), i);
}

Value ContractionOp::getSymbol(unsigned i) {
  return *std::next(operands().begin(), getNumTensors() + i);
}

void printContractionOp(OpAsmPrinter *printer, ContractionOp op) {
  SmallVector<StringRef, 3> elidedAttrs = {"agg", "combo", "name"};
  *printer << op.getOperation()->getName() << ' ';
  *printer << util::stringifyAggregationKind(op.agg());
  *printer << ", ";
  *printer << util::stringifyCombinationKind(op.combo());
  *printer << ", ";
  printer->printOperand(op.init());
  auto numTensors = op.getNumTensors();
  for (unsigned i = 0; i < numTensors; i++) {
    *printer << ", ";
    printer->printOperand(op.getTensor(i));
  }
  auto numSymbols = op.getNumSymbols();
  if (numSymbols) {
    *printer << " [";
    for (unsigned i = 0; i < numSymbols; i++) {
      if (i) {
        *printer << ", ";
      }
      printer->printOperand(op.getSymbol(i));
    }
    *printer << ']';
  }
  printer->printOptionalAttrDict(op->getAttrs(), elidedAttrs);
  *printer << " : ";
  printer->printType(op.init().getType());
  *printer << ", ";
  for (unsigned i = 0; i < numTensors; i++) {
    if (i) {
      *printer << ", ";
    }
    printer->printType(op.getTensor(i).getType());
  }
  *printer << " -> ";
  printer->printType(op.result().getType());
}

ParseResult parseContractionOp(OpAsmParser *parser, OperationState &result) {
  StringRef strAgg;
  StringRef strCombo;
  OpAsmParser::OperandType init;
  SmallVector<OpAsmParser::OperandType, 3> tensors;
  SmallVector<OpAsmParser::OperandType, 8> symbols;
  SmallVector<Type, 4> types;
  Type resultType;
  if (parser->parseKeyword(&strAgg) || parser->parseComma() ||
      parser->parseKeyword(&strCombo) || parser->parseComma() ||
      parser->parseOperand(init) || parser->parseComma()) {
    return failure();
  }

  auto agg = util::symbolizeAggregationKind(strAgg);
  if (!agg) {
    return failure();
  }
  result.addAttribute("agg", parser->getBuilder().getI64IntegerAttr(
                                 static_cast<int64_t>(agg.getValue())));

  auto combo = util::symbolizeCombinationKind(strCombo);
  if (!combo) {
    return failure();
  }
  result.addAttribute("combo", parser->getBuilder().getI64IntegerAttr(
                                   static_cast<int64_t>(combo.getValue())));

  auto numTensors = ContractionOp::getNumTensors(combo.getValue());
  if (parser->parseOperandList(tensors, numTensors) ||
      parser->parseOperandList(symbols,
                               OpAsmParser::Delimiter::OptionalSquare) ||
      parser->parseOptionalAttrDict(result.attributes) ||
      parser->parseColonTypeList(types) || parser->parseArrow() ||
      parser->parseType(resultType)) {
    return failure();
  }

  // TODO: parse a FunctionType here

  auto loc = parser->getCurrentLocation();
  auto indexType = parser->getBuilder().getIndexType();
  auto tensorTypes = llvm::makeArrayRef(types).drop_front();
  if (parser->resolveOperand(init, types.front(), result.operands) ||
      parser->resolveOperands(tensors, tensorTypes, loc, result.operands) ||
      parser->resolveOperands(symbols, indexType, result.operands)) {
    return failure();
  }

  result.addTypes(resultType);
  return success();
}

bool isAnyScalar(Type type) {
  return type.isIndex() || type.isa<FloatType>() || type.isInteger(1) ||
         type.isSignedInteger() || type.isUnsignedInteger();
}

bool isEltwiseAny(Type type) {
  if (auto rankedTensorType = type.dyn_cast<RankedTensorType>()) {
    auto elementType = rankedTensorType.getElementType();
    return isAnyScalar(elementType);
  }
  return isAnyScalar(type);
}

LogicalResult verifyContractionOp(ContractionOp op) {
  auto numTensors = op.getNumTensors();
  auto numSymbols = op.getNumSymbols();
  SmallVector<Value, 8> variadic(op.operands());
  if (variadic.size() < numTensors) {
    return op.emitOpError("combo '")
           << util::stringifyCombinationKind(op.combo()) << "' requires "
           << numTensors << " tensor operands";
  }
  auto shape = op.shape();
  auto resultType = op.result().getType().cast<RankedTensorType>();
  if (!resultType.hasStaticShape() && !shape.hasValue()) {
    return op.emitOpError(
        "attribute 'shape' is required when result type is dynamic");
  }
  unsigned expectedSymbols = op.sink().getNumSymbols();
  if (shape.hasValue()) {
    expectedSymbols += shape->getNumSymbols();
  }
  for (auto src : op.srcs()) {
    auto map = src.cast<AffineMapAttr>();
    expectedSymbols += map.getValue().getNumSymbols();
  }
  if (op.cons().hasValue()) {
    expectedSymbols += op.cons().getValue().getNumSymbols();
  }
  if (expectedSymbols != numSymbols) {
    return op.emitOpError("has incorrect number of symbols: expected ")
           << expectedSymbols << " but found " << numSymbols;
  }
  for (unsigned i = 0; i < numTensors; i++) {
    auto type = op.getTensor(i).getType();
    if (!isEltwiseAny(type)) {
      return op.emitOpError("tensor #")
             << i << " must be eltwise-any, but got " << type;
    }
  }
  for (unsigned i = 0; i < numSymbols; i++) {
    auto type = op.getSymbol(i).getType();
    if (!type.isa<IndexType>()) {
      return op.emitOpError("symbol #")
             << i << " must be index, but got " << type;
    }
  }
  return success();
}

LogicalResult verifyReshapeOp(ReshapeOp op) {
  auto inType = op.tensor().getType().cast<RankedTensorType>();
  auto outType = op.result().getType().cast<RankedTensorType>();
  if (inType.getElementType() != outType.getElementType()) {
    return op.emitOpError("element type mismatch");
  }
  if (inType.getNumElements() != outType.getNumElements()) {
    return op.emitOpError("element count mismatch");
  }
  return success();
}

void GatherOp::build(OpBuilder &builder, OperationState &result,
                     Type resultType, ValueRange operands, IntegerAttr axis,
                     IntegerAttr interpolationMode, IntegerAttr nearestMode,
                     FloatAttr cubeCoeff, IntegerAttr mode,
                     IntegerAttr batchDims) {
  assert(operands.size() == 2u && "mismatched number of parameters");
  result.addOperands(operands);
  result.addAttribute("axis", axis);
  result.addAttribute("interpolationMode", interpolationMode);
  result.addAttribute("nearestMode", nearestMode);
  result.addAttribute("cubeCoeff", cubeCoeff);
  result.addAttribute("mode", mode);
  result.addAttribute("batchDims", batchDims);
  result.addTypes(resultType);
}

void ScatterOp::build(OpBuilder &builder, OperationState &result,
                      Type resultType, ValueRange operands, IntegerAttr axis,
                      IntegerAttr mode) {
  assert(operands.size() == 3u && "mismatched number of parameters");
  result.addOperands(operands);
  result.addAttribute("axis", axis);
  result.addAttribute("mode", mode);
  result.addTypes(resultType);
}

//===----------------------------------------------------------------------===//
// LoopOp
//===----------------------------------------------------------------------===//

static LogicalResult verify(LoopOp op) {
  if (auto cst = op.step().getDefiningOp<mlir::ConstantIndexOp>())
    if (cst.getValue() <= 0)
      return op.emitOpError("constant step operand must be positive");

  // Check that the body defines as single block argument for the induction
  // variable.
  auto *body = op.getBody();
  if (!body->getArgument(0).getType().isIndex())
    return op.emitOpError(
        "expected body first argument to be an index argument for "
        "the induction variable");

  auto opNumResults = op.getNumResults();
  if (opNumResults == 0)
    return success();
  // If ForOp defines values, check that the number and types of
  // the defined values match ForOp initial iter operands and backedge
  // basic block arguments.
  if (op.getNumIterOperands() != opNumResults)
    return op.emitOpError(
        "mismatch in number of loop-carried values and defined values");
  if (op.getNumRegionIterArgs() != opNumResults)
    return op.emitOpError(
        "mismatch in number of basic block args and defined values");
  auto iterOperands = op.getIterOperands();
  auto iterArgs = op.getRegionIterArgs();
  auto opResults = op.getResults();
  unsigned i = 0;
  for (auto e : llvm::zip(iterOperands, iterArgs, opResults)) {
    if (std::get<0>(e).getType() != std::get<2>(e).getType())
      return op.emitOpError() << "types mismatch between " << i
                              << "th iter operand and defined value";
    if (std::get<1>(e).getType() != std::get<2>(e).getType())
      return op.emitOpError() << "types mismatch between " << i
                              << "th iter region arg and defined value";

    i++;
  }

  return RegionBranchOpInterface::verifyTypes(op);
}

/// Prints the initialization list in the form of
///   <prefix>(%inner = %outer, %inner2 = %outer2, <...>)
/// where 'inner' values are assumed to be region arguments and 'outer' values
/// are regular SSA values.
static void printInitializationList(OpAsmPrinter &p,
                                    Block::BlockArgListType blocksArgs,
                                    ValueRange initializers,
                                    StringRef prefix = "") {
  assert(blocksArgs.size() == initializers.size() &&
         "expected same length of arguments and initializers");
  if (initializers.empty())
    return;

  p << prefix << '(';
  llvm::interleaveComma(llvm::zip(blocksArgs, initializers), p, [&](auto it) {
    p << std::get<0>(it) << " = " << std::get<1>(it);
  });
  p << ")";
}

static void print(OpAsmPrinter &p, LoopOp op) {
  p << op.getOperationName() << " " << op.getInductionVar() << " = "
    << op.lowerBound() << " to " << op.upperBound() << " step " << op.step();

  printInitializationList(p, op.getRegionIterArgs(), op.getIterOperands(),
                          " iter_args");
  if (!op.getIterOperands().empty())
    p << " -> (" << op.getIterOperands().getTypes() << ')';
  p.printRegion(op.region(),
                /*printEntryBlockArgs=*/false,
                /*printBlockTerminators=*/op.hasIterOperands());
  p.printOptionalAttrDict(op.getAttrs());
}

static ParseResult parseLoopOp(OpAsmParser &parser, OperationState &result) {
  auto &builder = parser.getBuilder();
  OpAsmParser::OperandType inductionVariable, lb, ub, step;
  // Parse the induction variable followed by '='.
  if (parser.parseRegionArgument(inductionVariable) || parser.parseEqual())
    return failure();

  // Parse loop bounds.
  Type indexType = builder.getIndexType();
  if (parser.parseOperand(lb) ||
      parser.resolveOperand(lb, indexType, result.operands) ||
      parser.parseKeyword("to") || parser.parseOperand(ub) ||
      parser.resolveOperand(ub, indexType, result.operands) ||
      parser.parseKeyword("step") || parser.parseOperand(step) ||
      parser.resolveOperand(step, indexType, result.operands))
    return failure();

  // Parse the optional initial iteration arguments.
  SmallVector<OpAsmParser::OperandType, 4> regionArgs, operands;
  SmallVector<Type, 4> argTypes;
  regionArgs.push_back(inductionVariable);

  if (succeeded(parser.parseOptionalKeyword("iter_args"))) {
    // Parse assignment list and results type list.
    if (parser.parseAssignmentList(regionArgs, operands) ||
        parser.parseArrowTypeList(result.types))
      return failure();
    // Resolve input operands.
    for (auto operand_type : llvm::zip(operands, result.types))
      if (parser.resolveOperand(std::get<0>(operand_type),
                                std::get<1>(operand_type), result.operands))
        return failure();
  }
  // Induction variable.
  argTypes.push_back(indexType);
  // Loop carried variables
  argTypes.append(result.types.begin(), result.types.end());
  // Parse the body region.
  Region *body = result.addRegion();
  if (regionArgs.size() != argTypes.size())
    return parser.emitError(
        parser.getNameLoc(),
        "mismatch in number of loop-carried values and defined values");

  if (parser.parseRegion(*body, regionArgs, argTypes))
    return failure();

  LoopOp::ensureTerminator(*body, builder, result.location);

  // Parse the optional attribute list.
  if (parser.parseOptionalAttrDict(result.attributes))
    return failure();

  return success();
}
void LoopOp::build(OpBuilder &builder, OperationState &result, Value lb,
                   Value ub, Value step, ValueRange iterArgs,
                   BodyBuilderFn bodyBuilder) {
  result.addOperands({lb, ub, step});
  result.addOperands(iterArgs);
  for (Value v : iterArgs)
    result.addTypes(v.getType());
  Region *bodyRegion = result.addRegion();
  bodyRegion->push_back(new Block);
  Block &bodyBlock = bodyRegion->front();
  bodyBlock.addArgument(builder.getIndexType());
  for (Value v : iterArgs)
    bodyBlock.addArgument(v.getType());

  // Create the default terminator if the builder is not provided and if the
  // iteration arguments are not provided. Otherwise, leave this to the caller
  // because we don't know which values to return from the loop.
  if (iterArgs.empty() && !bodyBuilder) {
    LoopOp::ensureTerminator(*bodyRegion, builder, result.location);
  } else if (bodyBuilder) {
    OpBuilder::InsertionGuard guard(builder);
    builder.setInsertionPointToStart(&bodyBlock);
    bodyBuilder(builder, result.location, bodyBlock.getArgument(0),
                bodyBlock.getArguments().drop_front());
  }
}
Region &LoopOp::getLoopBody() { return region(); }

bool LoopOp::isDefinedOutsideOfLoop(Value value) {
  return !region().isAncestor(value.getParentRegion());
}

LogicalResult LoopOp::moveOutOfLoop(ArrayRef<Operation *> ops) {
  for (auto op : ops)
    op->moveBefore(*this);
  return success();
}

LoopOp getForInductionVarOwner(Value val) {
  auto ivArg = val.dyn_cast<BlockArgument>();
  if (!ivArg)
    return LoopOp();
  assert(ivArg.getOwner() && "unlinked block argument");
  auto *containingOp = ivArg.getOwner()->getParentOp();
  return dyn_cast_or_null<LoopOp>(containingOp);
}

/// Return operands used when entering the region at 'index'. These operands
/// correspond to the loop iterator operands, i.e., those exclusing the
/// induction variable. LoopOp only has one region, so 0 is the only valid value
/// for `index`.
OperandRange LoopOp::getSuccessorEntryOperands(unsigned index) {
  assert(index == 0 && "invalid region index");

  // The initial operands map to the loop arguments after the induction
  // variable.
  return initArgs();
}

/// Given the region at `index`, or the parent operation if `index` is None,
/// return the successor regions. These are the regions that may be selected
/// during the flow of control. `operands` is a set of optional attributes that
/// correspond to a constant value for each operand, or null if that operand is
/// not a constant.
void LoopOp::getSuccessorRegions(Optional<unsigned> index,
                                 ArrayRef<Attribute> operands,
                                 SmallVectorImpl<RegionSuccessor> &regions) {
  // If the predecessor is the ForOp, branch into the body using the iterator
  // arguments.
  if (!index.hasValue()) {
    regions.push_back(RegionSuccessor(&getLoopBody(), getRegionIterArgs()));
    return;
  }

  // Otherwise, the loop may branch back to itself or the parent operation.
  assert(index.getValue() == 0 && "expected loop region");
  regions.push_back(RegionSuccessor(&getLoopBody(), getRegionIterArgs()));
  regions.push_back(RegionSuccessor(getResults()));
}

void LoopOp::getNumRegionInvocations(ArrayRef<Attribute> operands,
                                     SmallVectorImpl<int64_t> &countPerRegion) {
  assert(countPerRegion.empty());
  countPerRegion.resize(1);

  auto lb = operands[0].dyn_cast_or_null<IntegerAttr>();
  auto ub = operands[1].dyn_cast_or_null<IntegerAttr>();
  auto step = operands[2].dyn_cast_or_null<IntegerAttr>();

  // Loop bounds are not known statically.
  if (!lb || !ub || !step || step.getValue().getSExtValue() == 0) {
    countPerRegion[0] = -1;
    return;
  }

  countPerRegion[0] =
      ceilDiv(ub.getValue().getSExtValue() - lb.getValue().getSExtValue(),
              step.getValue().getSExtValue());
}

//===----------------------------------------------------------------------===//
// YieldOp
//===----------------------------------------------------------------------===//

static ParseResult parseYieldOp(OpAsmParser &parser, OperationState &result) {
  SmallVector<OpAsmParser::OperandType, 4> operands;
  SmallVector<Type, 4> types;
  llvm::SMLoc loc = parser.getCurrentLocation();
  // Parse variadic operands list, their types, and resolve operands to SSA
  // values.
  if (parser.parseOperandList(operands) ||
      parser.parseOptionalColonTypeList(types) ||
      parser.resolveOperands(operands, types, loc, result.operands))
    return failure();
  return success();
}

static void print(OpAsmPrinter &p, tile::YieldOp op) {
  p << op.getOperationName();
  if (op.getNumOperands() != 0)
    p << ' ' << op.getOperands() << " : " << op.getOperandTypes();
}

} // namespace pmlc::dialect::tile

#include "pmlc/dialect/tile/ir/enums.cc.inc"

#define GET_OP_CLASSES
#include "pmlc/dialect/tile/ir/ops.cc.inc"

#include "pmlc/dialect/tile/ir/interfaces.cc.inc"
