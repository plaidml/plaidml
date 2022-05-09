// Copyright 2021, Intel Corporation

#include "pmlc/conversion/linalg_to_pxa/pass_detail.h"

namespace pmlc::conversion::linalg_to_pxa {

using namespace mlir; // NOLINT

LinalgToPXATypeConverter::LinalgToPXATypeConverter() {
  addConversion([](FunctionType type) { return type; });
  addConversion([](FloatType type) { return type; });
  addConversion([](IntegerType type) { return type; });
  addConversion([](IndexType type) { return type; });
  addConversion([](MemRefType type) { return type; });
  addConversion([this](RankedTensorType type) {
    Type elementType = type.getElementType();
    Type newType = convertType(elementType);
    assert(newType && "could not convert type");
    return MemRefType::get(type.getShape(), newType);
  });
}

// Generate a linalg.generic. This function uses bodyBuilder to create the
// different generic op bodies.
linalg::GenericOp createGenericOp(OpBuilder &builder, Operation *locationOp,
                                  TypeRange outputTypes, ValueRange inputs,
                                  ValueRange outputs, unsigned numIdxs,
                                  ArrayRef<AffineMap> maps,
                                  ArrayRef<StringRef> iterTypes,
                                  GenericOpBodyBuilder bodyBuilder) {
  builder.setInsertionPoint(locationOp);
  SmallVector<Value, 1> inits;
  SmallVector<Location, 4> argLocations;

  // Some original ops use outputs as operands, and some ops return outputs
  if (outputs.empty()) {
    // For the ops only return outputs, we have to initialize a tensor for its
    // output.
    for (auto outputType : outputTypes) {
      auto shapedType = outputType.cast<ShapedType>();
      auto outputShape = shapedType.getShape();
      auto elemType = shapedType.getElementType();
      auto initOp = builder.create<linalg::InitTensorOp>(locationOp->getLoc(),
                                                         outputShape, elemType);
      inits.emplace_back(initOp.getResult());
    }
  } else {
    // For the ops take output as operands, we can simply reuse these output
    // operands as generic op's output operands.
    inits.insert(inits.end(), outputs.begin(), outputs.end());
  }

  auto genericOp =
      builder.create<linalg::GenericOp>(locationOp->getLoc(),
                                        /*resultTensorTypes=*/outputTypes,
                                        /*inputs=*/inputs,
                                        /*outputs=*/inits,
                                        /*indexingMaps=*/maps,
                                        /*iteratorTypes=*/iterTypes);

  // Arguments for the loop body
  SmallVector<Type, 4> argTypes;
  for (auto input : inputs) {
    argTypes.emplace_back(input.getType().cast<ShapedType>().getElementType());
    argLocations.emplace_back(input.getLoc());
  }
  for (auto outputType : outputTypes) {
    argTypes.emplace_back(outputType.cast<ShapedType>().getElementType());
    // TODO: Lorenzo find a better way to pass location info.
    argLocations.emplace_back(builder.getUnknownLoc());
  }

  Block &block = genericOp.region().emplaceBlock();
  block.addArguments(argTypes, argLocations);
  builder.setInsertionPointToStart(&block);
  // Call bodyBuilder to create the customized generic op body.
  bodyBuilder(builder, genericOp.getLoc(), inputs.size(), block.getArguments());
  return genericOp;
}

} // namespace pmlc::conversion::linalg_to_pxa
