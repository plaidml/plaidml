// Copyright 2019, Intel Corporation

#include "pmlc/dialect/eltwise/ir/util.h"

#include <algorithm>
#include <string>
#include <utility>
#include <vector>

#include "llvm/Support/FormatVariadic.h"

#include "mlir/IR/Attributes.h"
#include "mlir/IR/Function.h"
#include "mlir/IR/Matchers.h"
#include "mlir/IR/StandardTypes.h"
#include "mlir/Support/DebugStringHelper.h"

#include "pmlc/util/logging.h"

namespace pmlc::dialect::eltwise {

using llvm::ArrayRef;
using llvm::SmallVector;
using mlir::Attribute;
using mlir::debugString;
using mlir::FloatAttr;
using mlir::FuncOp;
using mlir::IndexType;
using mlir::IntegerAttr;
using mlir::IntegerType;
using mlir::m_Constant;
using mlir::Operation;
using mlir::RankedTensorType;
using mlir::Type;
using mlir::Value;
using mlir::ValueRange;

namespace {

bool MergeTypes(Type *into, Type from, Type dtype) {
  IVLOG(6, "MergeTypes> " << debugString(*into) << ", " << debugString(from));
  auto intoShapedType = getRankedTensorType(*into);
  auto fromShapedType = getRankedTensorType(from);

  // To compute the result broadcasted shape, we compare operand shapes
  // element-wise: starting with the trailing dimensions, and working the
  // way backward. Two dimensions are compatible when
  //   1. they are equal, or
  //   2. one of them is 1
  // The result shape has the maximum among the two inputs at every
  // dimension index.
  SmallVector<int64_t, 4> resultShape;
  auto shape1 = intoShapedType.getShape();
  auto shape2 = fromShapedType.getShape();
  IVLOG(6, "  Checking compatibility between " << shape1.vec() << " and "
                                               << shape2.vec());
  if (shape1.size() > shape2.size()) {
    std::copy(shape1.begin(), shape1.end(), std::back_inserter(resultShape));
  } else {
    std::copy(shape2.begin(), shape2.end(), std::back_inserter(resultShape));
  }

  auto i1 = shape1.rbegin(), e1 = shape1.rend();
  auto i2 = shape2.rbegin(), e2 = shape2.rend();
  auto iR = resultShape.rbegin();

  // Check each dimension is consistent.
  for (; i1 != e1 && i2 != e2; ++i1, ++i2, ++iR) {
    if (*i1 == -1 || *i2 == -1) {
      // One or both dimensions is unknown. Follow TensorFlow behavior:
      // - If either dimension is greater than 1, we assume that the program is
      //   correct, and the other dimension will be broadcast to match it.
      // - If either dimension is 1, the other dimension is the output.
      if (*i1 > 1) {
        *iR = *i1;
      } else if (*i2 > 1) {
        *iR = *i2;
      } else if (*i1 == 1) {
        *iR = *i2;
      } else if (*i2 == 1) {
        *iR = *i1;
      } else {
        *iR = -1;
      }
    } else {
      if (*i1 == *i2 || *i2 == 1) {
        *iR = *i1;
      } else if (*i1 == 1) {
        *iR = *i2;
      } else {
        // This dimension of the two operand types is incompatible.
        return false;
      }
    }
  }

  if (!dtype) {
    auto intoElementType = intoShapedType.getElementType();
    auto fromElementType = fromShapedType.getElementType();
    dtype = promoteTypes(intoElementType, fromElementType);
  }
  *into = RankedTensorType::get(resultShape, dtype);
  IVLOG(6, "  Resulting type: " << debugString(*into));
  return true;
}

} // namespace

// I64EnumAttrCase<"invalid", 0>,
// I64EnumAttrCase<"u1",      1>,
// I64EnumAttrCase<"i8",      2>,
// I64EnumAttrCase<"u8",      3>,
// I64EnumAttrCase<"i16",     4>,
// I64EnumAttrCase<"u16",     5>,
// I64EnumAttrCase<"i32",     6>,
// I64EnumAttrCase<"u32",     7>,
// I64EnumAttrCase<"i64",     8>,
// I64EnumAttrCase<"u64",     9>,
// I64EnumAttrCase<"bf16",   10>,
// I64EnumAttrCase<"f16",    11>,
// I64EnumAttrCase<"f32",    12>,
// I64EnumAttrCase<"f64",    13>,
unsigned typeScore(Type type) {
  if (!type || type.isIndex()) {
    return 0;
  }
  if (type.isInteger(1)) {
    return 1;
  }
  if (type.isSignedInteger(8)) {
    return 2;
  }
  if (type.isUnsignedInteger(8)) {
    return 3;
  }
  if (type.isSignedInteger(16)) {
    return 4;
  }
  if (type.isUnsignedInteger(16)) {
    return 5;
  }
  if (type.isSignedInteger(32)) {
    return 6;
  }
  if (type.isUnsignedInteger(32)) {
    return 7;
  }
  if (type.isSignedInteger(64)) {
    return 8;
  }
  if (type.isUnsignedInteger(64)) {
    return 9;
  }
  if (type.isBF16()) {
    return 10;
  }
  if (type.isF16()) {
    return 11;
  }
  if (type.isF32()) {
    return 12;
  }
  if (type.isF64()) {
    return 13;
  }
  assert(false && "Undefined typeScore");
  return 0;
}

Type promoteTypes(Type lhs, Type rhs) {
  return typeScore(lhs) > typeScore(rhs) ? lhs : rhs;
}

RankedTensorType getRankedTensorType(Type type) {
  if (auto rankedType = type.dyn_cast<RankedTensorType>()) {
    return rankedType;
  }
  SmallVector<int64_t, 0> shape;
  if (type.isa<IndexType>()) {
    // TODO: reify this when we lower to a specific target.
    return RankedTensorType::get(
        shape, IntegerType::get(32, IntegerType::SignednessSemantics::Signed,
                                type.getContext()));
  }
  return RankedTensorType::get(shape, type);
}

Type ComputeResultType(ValueRange operands, Type override) {
  if (VLOG_IS_ON(6)) {
    std::vector<std::string> types;
    for (auto operand : operands) {
      auto type = operand.getType();
      types.push_back(debugString(type));
    }
    IVLOG(6, "ComputeResultType> " << types);
  }
  Type ret = (*operands.begin()).getType();
  for (auto operand : operands.drop_front()) {
    auto type = operand.getType();
    if (!MergeTypes(&ret, type, override)) {
      std::stringstream ss;
      ss << "Incompatible types: (";
      for (size_t i = 0; i < operands.size(); i++) {
        if (i) {
          ss << ", ";
        }
        auto type = operands[i].getType();
        ss << debugString(type);
      }
      ss << ")";
      throw std::runtime_error(ss.str());
    }
  }
  return ret;
}

SmallVector<int64_t, 4> ComputeShape(ArrayRef<Value> operands) {
  SmallVector<int64_t, 4> shape;
  for (auto operand : operands) {
    auto op = operand.getDefiningOp();
    IntegerAttr attr;
    if (m_Constant(&attr).match(op)) {
      shape.push_back(attr.getInt());
    } else {
      shape.push_back(-1);
    }
  }
  return shape;
}

Attribute constFoldUnaryOp(ArrayRef<Attribute> operands,
                           UnaryCalculate calculate) {
  assert(operands.size() == 1 && "unary op takes one operand");
  if (auto op = operands[0].dyn_cast_or_null<FloatAttr>()) {
    return FloatAttr::get(op.getType(), calculate(op.getValueAsDouble()));
  }
  if (auto op = operands[0].dyn_cast_or_null<IntegerAttr>()) {
    return IntegerAttr::get(op.getType(), calculate(op.getInt()));
  }
  return {};
}

Attribute constFoldBinaryOp(ArrayRef<Attribute> operands,
                            BinaryCalculate calculate) {
  assert(operands.size() == 2 && "binary op takes two operands");
  if (auto lhs = operands[0].dyn_cast_or_null<FloatAttr>()) {
    if (auto rhs = operands[1].dyn_cast_or_null<FloatAttr>()) {
      return FloatAttr::get(lhs.getType(), calculate(lhs.getValueAsDouble(),
                                                     rhs.getValueAsDouble()));
    }
    if (auto rhs = operands[1].dyn_cast_or_null<IntegerAttr>()) {
      return FloatAttr::get(lhs.getType(),
                            calculate(lhs.getValueAsDouble(), rhs.getInt()));
    }
  } else if (auto lhs = operands[0].dyn_cast_or_null<IntegerAttr>()) {
    if (auto rhs = operands[1].dyn_cast_or_null<FloatAttr>()) {
      return FloatAttr::get(rhs.getType(),
                            calculate(lhs.getInt(), rhs.getValueAsDouble()));
    }
    if (auto rhs = operands[1].dyn_cast_or_null<IntegerAttr>()) {
      return IntegerAttr::get(lhs.getType(),
                              calculate(lhs.getInt(), rhs.getInt()));
    }
  }
  return {};
}

bool ConstantValueMatcher::match(Operation *op) {
  Attribute attr;
  if (!mlir::detail::constant_op_binder<Attribute>(&attr).match(op)) {
    return false;
  }
  if (auto intAttr = attr.dyn_cast<IntegerAttr>()) {
    return intAttr.getValue() == value;
  }
  if (auto floatAttr = attr.dyn_cast<FloatAttr>()) {
    return floatAttr.getValueAsDouble() == value;
  }
  return false;
}

Type toSignlessType(Type type) {
  if (auto integerType = type.dyn_cast<IntegerType>()) {
    return IntegerType::get(integerType.getWidth(), type.getContext());
  }
  return type;
}

} // namespace pmlc::dialect::eltwise
