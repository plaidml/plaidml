// Copyright 2020 Intel Corporation

#include "pmlc/ast/ast_ops.h"

#include "llvm/ADT/Optional.h"

#include "pmlc/ast/ast.h"
#include "pmlc/ast/eval.h"
#include "pmlc/util/logging.h"

using llvm::ArrayRef;

namespace pmlc::ast {

using util::DataType;
using util::TensorShape;
using util::TensorShapes;

static llvm::Optional<int64_t> getIntegerValue(Evaluator *evaluator,
                                               const ExprNodePtr &operand) {
  if (auto node = std::dynamic_pointer_cast<ExprNodeConstSigned>(operand)) {
    return node->value;
  }
  if (auto node = std::dynamic_pointer_cast<ExprNodeConstUnsigned>(operand)) {
    return node->value;
  }
  if (auto node = std::dynamic_pointer_cast<ExprNodeDim>(operand)) {
    return evaluator->evaluate(node->dim);
  }
  return llvm::None;
}

static llvm::Optional<double> getFloatValue(Evaluator *evaluator,
                                            const ExprNodePtr &operand) {
  if (auto node = std::dynamic_pointer_cast<ExprNodeConstFloat>(operand)) {
    return node->value;
  }
  return llvm::None;
}

static bool isDataTypeFloat(DataType type) {
  return type == DataType::f16 || type == DataType::f32 ||
         type == DataType::f64;
}

static bool isDataTypeInteger(DataType type) {
  return type == DataType::i1 || type == DataType::si8 ||
         type == DataType::ui8 || type == DataType::si16 ||
         type == DataType::ui16 || type == DataType::si32 ||
         type == DataType::ui32 || type == DataType::si64 ||
         type == DataType::ui64;
}

struct ArgSortOp : Intrinsic {
  TensorShapes getShapes(Evaluator *evaluator, ArrayRef<ExprNodePtr> operands,
                         ArrayRef<TensorShape> shapes) const final {
    if (operands.size() != 3) {
      throw std::runtime_error("'argsort' requires 3 arguments.");
    }
    auto axis = getIntegerValue(evaluator, operands[1]);
    if (!axis) {
      throw std::runtime_error(
          "'argsort' requires operand #2 to be an integer.");
    }
    auto direction = getIntegerValue(evaluator, operands[2]);
    if (!direction) {
      throw std::runtime_error(
          "'argsort' requires operand #3 to be an integer.");
    }
    return {TensorShape(DataType::si32, shapes[0].sizes)};
  }
};

struct BooleanOp : Intrinsic {
  TensorShapes getShapes(Evaluator *evaluator, ArrayRef<ExprNodePtr> operands,
                         ArrayRef<TensorShape> shapes) const final {
    return {inferShape(shapes, /*override=*/DataType::i1)};
  }
};

struct IndexOp : Intrinsic {
  TensorShapes getShapes(Evaluator *evaluator, ArrayRef<ExprNodePtr> operands,
                         ArrayRef<TensorShape> shapes) const final {
    if (operands.size() < 2) {
      throw std::runtime_error(
          "'index' requires an axis operand and at least 1 dimension");
    }
    TensorShape ret(DataType::si32); // TODO
    for (const ExprNodePtr &operand : operands.drop_front()) {
      auto value = getIntegerValue(evaluator, operand);
      if (!value) {
        throw std::runtime_error("Additional parameters to 'index' must be "
                                 "integers or TensorDims.");
      }
      ret.sizes.push_back(*value);
    }
    return {ret};
  }
};

struct GatherOp : Intrinsic {
  TensorShapes getShapes(Evaluator *evaluator, ArrayRef<ExprNodePtr> operands,
                         ArrayRef<TensorShape> shapes) const final {
    auto operands_size = operands.size();
    if (operands_size != 8) {
      throw std::runtime_error("'gather' requires eight arguments.");
    }
    auto tensor = shapes[0];
    auto idxs = shapes[1];
    if (isDataTypeFloat(idxs.elementType) &&
        !isDataTypeFloat(tensor.elementType)) {
      throw std::runtime_error("'gather' interpolation modes require tensor "
                               "elements to be floats.");
    }
    int64_t rank = tensor.getRank();
    if (!rank) {
      throw std::runtime_error(
          "'gather' requires first operand to have at least one dimension.");
    }
    auto axis = getIntegerValue(evaluator, operands[2]);
    if (!axis || axis.getValue() >= rank || axis.getValue() < 0) {
      throw std::runtime_error(
          "'gather' primitive expects the 'axis' argument "
          "to be a positive integer that is less than the tensor rank.");
    }
    auto interpolationMode = getIntegerValue(evaluator, operands[3]);
    if (!interpolationMode) {
      throw std::runtime_error(
          "'gather' primitive expects the 'interpolationMode' argument "
          "to be a constant integer");
    }
    auto nearestMode = getIntegerValue(evaluator, operands[4]);
    if (!nearestMode) {
      throw std::runtime_error(
          "'gather' primitive expects the 'nearestMode' argument "
          "to be a constant integer");
    }
    auto cubeCoeff = getFloatValue(evaluator, operands[5]);
    if (!cubeCoeff) {
      throw std::runtime_error(
          "'gather' primitive expects the 'cubeCoeff' argument "
          "to be a constant float");
    }

    auto mode = getIntegerValue(evaluator, operands[6]);
    if (!mode) {
      throw std::runtime_error("'gather' primitive expects the 'mode' argument "
                               "to be a constant integer");
    } else if (mode.getValue() == 1) {
      if (!isDataTypeInteger(idxs.elementType)) {
        throw std::runtime_error(
            "'gather' ND mode requires indices elements to be integers.");
      }
    }

    auto batchDims = getIntegerValue(evaluator, operands[7]);
    if (!batchDims) {
      throw std::runtime_error(
          "'gather' primitive expects the 'batchDims' argument "
          "to be a constant integer");
    }

    TensorShape shape{tensor.elementType};
    if (mode.getValue() == 0) {
      for (auto i = 0; i < axis.getValue(); i++) {
        shape.sizes.push_back(tensor.sizes[i]);
      }
      shape.sizes.insert(shape.sizes.end(), idxs.sizes.begin(),
                         idxs.sizes.end());
      for (auto i = axis.getValue() + 1; i < rank; i++) {
        shape.sizes.push_back(tensor.sizes[i]);
      }
    } else {
      for (size_t i = 0; i < idxs.getRank() - 1; i++) {
        shape.sizes.push_back(idxs.sizes[i]);
      }
      for (auto i = idxs.sizes.back() + batchDims.getValue(); i < rank; i++) {
        shape.sizes.push_back(tensor.sizes[i]);
      }
    }
    return {shape};
  }
};

struct PrngOp : Intrinsic {
  TensorShapes getShapes(Evaluator *evaluator, ArrayRef<ExprNodePtr> operands,
                         ArrayRef<TensorShape> shapes) const final {
    if (operands.size() < 1) {
      throw std::runtime_error("'prng' requires at least one argument.");
    }
    std::vector<int64_t> dims;
    for (const ExprNodePtr &operand : operands.drop_front()) {
      auto value = getIntegerValue(evaluator, operand);
      if (!value) {
        throw std::runtime_error("'prng' has invalid operands, dims must be "
                                 "tensor dimensions");
      }
      dims.push_back(*value);
    }
    return {TensorShape(DataType::f32, dims), shapes[0]};
  }
};

struct ReshapeOp : Intrinsic {
  TensorShapes getShapes(Evaluator *evaluator, ArrayRef<ExprNodePtr> operands,
                         ArrayRef<TensorShape> shapes) const final {
    if (operands.size() < 1) {
      throw std::runtime_error("'reshape' requires at least one argument.");
    }
    TensorShape ret(shapes[0].elementType);
    for (const ExprNodePtr &operand : operands.drop_front()) {
      auto value = getIntegerValue(evaluator, operand);
      if (!value) {
        throw std::runtime_error("Additional parameters to 'reshape' must be "
                                 "integers or TensorDims.");
      }
      ret.sizes.push_back(*value);
    }
    return {ret};
  }
};

struct ScatterOp : Intrinsic {
  TensorShapes getShapes(Evaluator *evaluator, ArrayRef<ExprNodePtr> operands,
                         ArrayRef<TensorShape> shapes) const final {
    if (operands.size() != 5) {
      throw std::runtime_error("'scatter' requires 5 operands.");
    }
    // data tensor
    int64_t rank = shapes[0].getRank();
    if (!rank) {
      throw std::runtime_error(
          "'scatter' requires operand #1 to have at least one dimension.");
    }
    // indices tensor
    if (shapes[1].elementType != DataType::si32 &&
        shapes[1].elementType != DataType::ui32) {
      // TODO: Handle other integer types?  Floor floats?
      throw std::runtime_error(
          "'scatter' requires operand #2 to be an integer.");
    }

    auto axis = getIntegerValue(evaluator, operands[3]);
    if (!axis || axis.getValue() >= rank) {
      throw std::runtime_error(
          "'scatter' primitive expects the 'axis' argument "
          "to be a positive integer that is less than the updates tensor "
          "rank.");
    }

    auto mode = getIntegerValue(evaluator, operands[4]);
    if (!mode) {
      throw std::runtime_error(
          "'scatter' primitive expects the 'mode' argument "
          "to be a constant integer.");
    }

    TensorShape ret(shapes[0].elementType, shapes[0].sizes);
    return {ret};
  }
};

struct SelectOp : Intrinsic {
  TensorShapes getShapes(Evaluator *evaluator, ArrayRef<ExprNodePtr> operands,
                         ArrayRef<TensorShape> shapes) const final {
    TensorShape shape = inferShape(shapes);
    DataType elementType = inferElementType(shapes.drop_front());
    if (isAmbiguousDataType(elementType)) {
      throw std::runtime_error(
          "'select' has ambiguous operand types, use a cast");
    }
    for (const TensorShape &shape : shapes.drop_front()) {
      if (shape.elementType != elementType) {
        throw std::runtime_error(
            "'select' has unmatched operand types, use a cast");
      }
    }
    return {TensorShape(elementType, shape.sizes)};
  }
};

struct ShapeOp : Intrinsic {
  TensorShapes getShapes(Evaluator *evaluator, ArrayRef<ExprNodePtr> operands,
                         ArrayRef<TensorShape> shapes) const final {
    if (operands.size() != 1) {
      throw std::runtime_error("'shape' requires exactly one argument.");
    }
    int64_t rank = shapes[0].getRank();
    return {TensorShape(DataType::si32, {rank})};
  }
};

void registerOps() {
  auto registry = IntrinsicRegistry::Instance();
  registry->add("argsort", std::make_unique<ArgSortOp>());
  registry->add("cmp_eq", std::make_unique<BooleanOp>());
  registry->add("cmp_ge", std::make_unique<BooleanOp>());
  registry->add("cmp_gt", std::make_unique<BooleanOp>());
  registry->add("cmp_le", std::make_unique<BooleanOp>());
  registry->add("cmp_lt", std::make_unique<BooleanOp>());
  registry->add("cmp_ne", std::make_unique<BooleanOp>());
  registry->add("gather", std::make_unique<GatherOp>());
  registry->add("index", std::make_unique<IndexOp>());
  registry->add("logical_and", std::make_unique<BooleanOp>());
  registry->add("logical_not", std::make_unique<BooleanOp>());
  registry->add("logical_or", std::make_unique<BooleanOp>());
  registry->add("logical_xor", std::make_unique<BooleanOp>());
  registry->add("prng", std::make_unique<PrngOp>());
  registry->add("reshape", std::make_unique<ReshapeOp>());
  registry->add("scatter", std::make_unique<ScatterOp>());
  registry->add("select", std::make_unique<SelectOp>());
  registry->add("shape", std::make_unique<ShapeOp>());
}

} // namespace pmlc::ast
