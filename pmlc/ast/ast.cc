// Copyright 2020 Intel Corporation

#include "pmlc/ast/ast.h"

#include <algorithm>
#include <sstream>

#include "llvm/Support/FormatVariadic.h"

#include "pmlc/util/logging.h"

namespace pmlc::ast {

using util::DataType;
using util::TensorShape;

static llvm::StringRef getAffineOpStr(AffineOp op) {
  switch (op) {
  case AffineOp::Add:
    return "add";
  case AffineOp::Div:
    return "div";
  case AffineOp::Max:
    return "max";
  case AffineOp::Min:
    return "min";
  case AffineOp::Mul:
    return "mul";
  case AffineOp::Neg:
    return "neg";
  case AffineOp::Sub:
    return "sub";
  default:
    return "<invalid op>";
  }
  llvm_unreachable("getAffineOpStr");
}

static int64_t getTypeScore(DataType type) {
  return static_cast<int64_t>(type);
}

static DataType promoteTypes(DataType lhs, DataType rhs) {
  return getTypeScore(lhs) > getTypeScore(rhs) ? lhs : rhs;
}

DataType inferElementType(llvm::ArrayRef<TensorShape> shapes) {
  DataType ret = DataType::invalid;
  for (const TensorShape &shape : shapes) {
    ret = promoteTypes(ret, shape.elementType);
  }
  return ret;
}

static bool mergeShapes(TensorShape *into, const TensorShape &from,
                        DataType dtype) {
  // To compute the resulting broadcasted shape, we compare operand shapes
  // element-wise: starting with the trailing dimensions, and working our
  // way backward. Two dimensions are compatible when
  //   1. they are equal, or
  //   2. one of them is 1
  // The result shape has the maximum among the two inputs at every
  // dimension index.
  std::vector<int64_t> resultShape;
  const std::vector<int64_t> &shape1 = into->sizes;
  const std::vector<int64_t> &shape2 = from.sizes;
  IVLOG(6, "  Checking compatibility between " << shape1 << " and " << shape2);
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
    if (*i1 == 0 || *i2 == 0) {
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
        *iR = 0;
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

  if (dtype == DataType::invalid) {
    dtype = promoteTypes(into->elementType, from.elementType);
  }
  *into = TensorShape{dtype, resultShape};
  IVLOG(6, "  Resulting shape: " << into->str());
  return true;
}

TensorShape inferShape(llvm::ArrayRef<TensorShape> operands,
                       DataType override) {
  TensorShape ret = operands.front();
  if (override != DataType::invalid) {
    ret.elementType = override;
  }
  for (const TensorShape &operand : operands.drop_front()) {
    if (!mergeShapes(&ret, operand, override)) {
      std::stringstream ss;
      ss << "Incompatible types: (";
      for (size_t i = 0; i < operands.size(); i++) {
        if (i) {
          ss << ", ";
        }
        ss << operands[i].str();
      }
      ss << ")";
      throw std::runtime_error(ss.str());
    }
  }
  return ret;
}

//
// TensorShape
//

//
// ExprNode
//

ExprNode::ExprNode(llvm::StringRef name) : name(name) {}

//
// ExprNodeCast
//

ExprNodeCast::ExprNodeCast(DataType dtype, const ExprNodePtr &expr)
    : dtype(dtype), expr(expr) {}

std::string ExprNodeCast::str() const { return "cast"; }

//
// ExprNodeConstSsigned
//

ExprNodeConstSigned::ExprNodeConstSigned(int64_t value) : value(value) {}

std::string ExprNodeConstSigned::str() const { return std::to_string(value); }

//
// ExprNodeConstUnsigned
//

ExprNodeConstUnsigned::ExprNodeConstUnsigned(uint64_t value) : value(value) {}

std::string ExprNodeConstUnsigned::str() const { return std::to_string(value); }

//
// ExprNodeConstFloat
//

ExprNodeConstFloat::ExprNodeConstFloat(double value) : value(value) {}

std::string ExprNodeConstFloat::str() const { return std::to_string(value); }

//
// ExprNodeConstTensor
//

ExprNodeConstTensor::ExprNodeConstTensor(const util::BufferPtr &buffer,
                                         llvm::StringRef name)
    : buffer(buffer) {}

std::string ExprNodeConstTensor::str() const { return "constant_tensor"; }

std::string Constraint::str() const {
  return llvm::formatv("{0} < {1}", lhs->str(), rhs->str());
}

//
// ExprNodeContraction
//

ExprNodeContraction::ExprNodeContraction(llvm::StringRef name) : Base(name) {}

std::string ExprNodeContraction::str() const { return "contraction"; }

//
// ExprNodeDim
//

ExprNodeDim::ExprNodeDim(const DimNodePtr &dim) : dim(dim) {}

std::string ExprNodeDim::str() const { return dim->str(); }

//
// ExprNodeElement
//

ExprNodeElement::ExprNodeElement(const ExprNodePtr &expr, size_t ordinal)
    : expr(expr), ordinal(ordinal) {}

std::string ExprNodeElement::str() const {
  return llvm::formatv("{0}[{1}]", expr->str(), ordinal);
}

//
// ExprNodeInput
//

ExprNodeInput::ExprNodeInput(const TensorShape &shape, llvm::StringRef name)
    : Base(name), shape(shape) {}

std::string ExprNodeInput::str() const {
  if (name.size()) {
    return llvm::formatv("input<{0}, \"{1}\">", shape.str(), name);
  }
  return llvm::formatv("input<{0}>", shape.str());
}

//
// ExprNodeIntrinsic
//

ExprNodeIntrinsic::ExprNodeIntrinsic(llvm::StringRef op,
                                     llvm::ArrayRef<ExprNodePtr> operands)
    : op(op), operands(operands) {}

std::string ExprNodeIntrinsic::str() const {
  return llvm::formatv("{0}()", op);
}

//
// ExprNodeTrace
//

ExprNodeTrace::ExprNodeTrace(const ExprNodePtr &expr, llvm::StringRef msg)
    : expr(expr), msg(msg) {}

std::string ExprNodeTrace::str() const {
  return llvm::formatv("trace(\"{0}\")", msg);
}

//
// DimNode tree
//

std::string DimNodeLiteral::str() const { return std::to_string(value); }

std::string DimNodeOp::str() const { return getAffineOpStr(op).str(); }

std::string DimNodeRef::str() const {
  return llvm::formatv("{0}[{1}]", ref->str(), dim);
}

//
// PolyNode tree
//

std::string PolyNodeDim::str() const { return dim->str(); }

std::string PolyNodeIndex::str() const { return llvm::formatv("%{0}", name); }

std::string PolyNodeLiteral::str() const { return std::to_string(value); }

std::string PolyNodeOp::str() const { return getAffineOpStr(op).str(); }

//
// VarNode tree
//

std::string VarNodeTuple::str() const {
  std::stringstream ss;
  ss << '(';
  for (auto item : llvm::enumerate(values)) {
    if (item.index()) {
      ss << ", ";
    }
    ss << item.value()->str();
  }
  ss << ')';
  return ss.str();
}

} // namespace pmlc::ast
