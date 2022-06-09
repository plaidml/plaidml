// Copyright 2021 Intel Corporation

#include "pmlc/dialect/linalgx/ir/ops.h"

#include "mlir/IR/DialectImplementation.h"
#include "mlir/IR/FunctionImplementation.h"
#include "mlir/IR/OpImplementation.h"

#include "pmlc/util/logging.h"

using namespace mlir; // NOLINT

namespace pmlc::dialect::linalgx {

/// Generic entry point to create the block for the region of a LinalgOp.
/// This is used by both named structured ops created by ods-gen and by manually
/// defined C++ ops.
/// This is used by both builders and parsers.
/// This function creates the block in the region with arguments corresponding
/// to the elemental types of `inputTypes` and `outputTypes`, which are asserted
/// to be ShapedType.
template <typename NamedStructuredOpType>
static void fillStructuredOpRegion(
    OpBuilder &opBuilder, Region &region, TypeRange inputTypes,
    TypeRange outputTypes,
    std::function<void(unsigned, unsigned)> errorHandler = nullptr) {
  assert(llvm::all_of(outputTypes, [](Type t) { return t.isa<ShapedType>(); }));

  // TODO: atm all operands go through getElementTypeOrSelf,
  // reconsider when we have evidence we need to.
  // TODO: find a way to pass proper location.
  SmallVector<Type, 8> argTypes;
  SmallVector<Location> locations;
  for (auto containers : {inputTypes, outputTypes})
    for (auto t : containers) {
      argTypes.push_back(getElementTypeOrSelf(t));
      locations.push_back(opBuilder.getUnknownLoc());
    }

  // RAII.
  OpBuilder::InsertionGuard guard(opBuilder);
  Block *body = opBuilder.createBlock(&region, /*insertPt=*/{}, argTypes, locations);
  unsigned actual = body->getNumArguments();
  unsigned expected = NamedStructuredOpType::getNumRegionArgs();
  if (expected != actual) {
    if (errorHandler)
      errorHandler(expected, actual);
    return;
  }

  opBuilder.setInsertionPointToStart(body);
  ImplicitLocOpBuilder b(opBuilder.getUnknownLoc(), opBuilder);
  NamedStructuredOpType::regionBuilder(b, *body);

  // indexing_maps is an auto-generated method.

  // iterator_types is an auto-generated method.
}

//===----------------------------------------------------------------------===//
// CopyOp
//===----------------------------------------------------------------------===//

void CopyOp::build(OpBuilder &builder, OperationState &result, Value input,
                   Value output, AffineMap inputMap, AffineMap outputMap) {
  result.addTypes({output.getType()});
  result.addOperands({input, output});
  if (inputMap)
    result.addAttribute("inputMap", AffineMapAttr::get(inputMap));
  if (outputMap)
    result.addAttribute("outputMap", AffineMapAttr::get(outputMap));
  result.addRegion();
  fillStructuredOpRegion<CopyOp>(builder, *result.regions.front(),
                                 TypeRange{input.getType()},
                                 TypeRange{output.getType()});
}

ArrayAttr CopyOp::indexing_maps() {
  MLIRContext *context = getContext();
  auto maybeInputMap = inputMap();
  auto maybeOutputMap = outputMap();
  int64_t inputRank = getRank(getInputOperand(0));
  int64_t outputRank = getRank(getOutputOperand(0));
  return Builder(getContext())
      .getAffineMapArrayAttr(
          {linalg::extractOrIdentityMap(maybeInputMap, inputRank, context),
           linalg::extractOrIdentityMap(maybeOutputMap, outputRank, context)});
}

ArrayAttr CopyOp::iterator_types() {
  int64_t numLoops = getTiedIndexingMap(getInputOperand(0)).getNumDims();
  return Builder(getContext())
      .getStrArrayAttr(
          SmallVector<StringRef, 8>(numLoops, getParallelIteratorTypeName()));
}

void CopyOp::regionBuilder(ImplicitLocOpBuilder &b, Block &block, llvm::ArrayRef<NamedAttribute> attrs = {}) {
  assert(block.getNumArguments() == 2 && "CopyOp regionBuilder expects 2 args");
  b.create<linalg::YieldOp>(block.getArgument(0));
}

ParseResult parseCopyOpRegion(OpAsmParser &parser, Region &region,
                              Type inputType, Type outputType) {
  OpBuilder opBuilder(parser.getBuilder().getContext());
  fillStructuredOpRegion<CopyOp>(opBuilder, region, TypeRange{inputType},
                                 TypeRange{outputType});
  return success();
}

/// CopyOp region is elided when printing.
void printCopyOpRegion(OpAsmPrinter &, Operation *, Region &, Type, Type) {}

LogicalResult CopyOp::verify() {
  OpOperand *output = getOutputOperand(0);
  OpOperand *input = getInputOperand(0);
  if (getElementTypeOrSelf(input->get()) != getElementTypeOrSelf(output->get()))
    return emitOpError("expects views of the same element type");
  // if (op.getRank(input) != op.getRank(output))
  //   return op.emitOpError("expects views of the same rank");
  auto rank = getNumParallelLoops();
  auto inMap = inputMap();
  if (inMap) {
    if (inMap->getNumInputs() != rank)
      return emitOpError("expects optional input_map of rank ") << rank;
    // if (!inputMap->isPermutation())
    //   return op.emitOpError(
    //       "expects optional input_map to be a permutation");
  }
  auto outMap = outputMap();
  if (outMap) {
    if (outMap->getNumInputs() != rank)
      return emitOpError("expects optional output_map of rank ") << rank;
    // if (!outputMap->isPermutation())
    //   return op.emitOpError(
    //       "expects optional output_map to be a permutation");
  }
  if (rank == 0 && inMap)
    return emitOpError("expected no input map when rank == 0");
  if (rank == 0 && outMap)
    return emitOpError("expected no output map when rank == 0");
  return success();
}

// ---- LinalgXDialect ----

void LinalgXDialect::initialize() {
  addOperations<
#define GET_OP_LIST
#include "pmlc/dialect/linalgx/ir/ops.cc.inc" // NOLINT
      >();
}

} // namespace pmlc::dialect::linalgx

#include "pmlc/dialect/linalgx/ir/dialect.cc.inc" // NOLINT

#define GET_OP_CLASSES
#include "pmlc/dialect/linalgx/ir/ops.cc.inc" // NOLINT
