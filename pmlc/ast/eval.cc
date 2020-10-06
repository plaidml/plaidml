// Copyright 2020 Intel Corporation

#include "pmlc/ast/eval.h"

#include <algorithm>

#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/TypeSwitch.h"

#include "pmlc/ast/ast_ops.h"
#include "pmlc/util/logging.h"

namespace pmlc::ast {

using util::DataType;
using util::TensorShape;
using util::TensorShapes;

static DataType inferElementType(Evaluator *evaluator,
                                 util::CombinationKind combo,
                                 llvm::ArrayRef<PolyMap> srcs) {
  if (combo == util::CombinationKind::eq) {
    return DataType::i1;
  }
  if (combo == util::CombinationKind::cond) {
    return evaluator->getShape(srcs[2].ref).elementType;
  }
  llvm::SmallVector<TensorShape, 3> shapes;
  for (const PolyMap &src : srcs) {
    shapes.push_back(evaluator->getShape(src.ref));
  }
  return ast::inferElementType(shapes);
}

bool isAmbiguousDataType(DataType dtype) {
  return dtype == DataType::fx || dtype == DataType::six ||
         dtype == DataType::uix;
}

//
// Evaluator
//

void Evaluator::verify(const ExprNodePtr &node) { (void)getShapes(node); }

int64_t Evaluator::evaluate(const DimNodePtr &node) {
  return evaluate(node.get());
}

int64_t Evaluator::evaluate(const DimNode *node) {
  return llvm::TypeSwitch<const DimNode *, int64_t>(node)
      .Case<DimNodeLiteral>([&](const auto *node) { return node->value; })
      .Case<DimNodeNone>([&](const auto *node) -> int64_t {
        throw std::runtime_error("DimNodeNone cannot be evaluated");
      })
      .Case<DimNodeOp>([&](const auto *node) {
        if (node->op == AffineOp::Neg) {
          if (node->operands.size() != 1) {
            throw std::runtime_error("AffineOp::Neg expected single operand");
          }
          return -evaluate(node->operands[0]);
        }
        if (node->operands.size() != 2) {
          throw std::runtime_error("Binary DimNodeOp expected 2 operands");
        }
        int64_t lhs = evaluate(node->operands[0]);
        int64_t rhs = evaluate(node->operands[1]);
        switch (node->op) {
        case AffineOp::Add:
          return lhs + rhs;
        case AffineOp::Sub:
          return lhs - rhs;
        case AffineOp::Mul:
          return lhs * rhs;
        case AffineOp::Div:
          return lhs / rhs;
        case AffineOp::Max:
          return std::max(lhs, rhs);
        case AffineOp::Min:
          return std::min(lhs, rhs);
        default:
          throw std::runtime_error("Invalid AffineOp");
        }
      })
      .Case<DimNodeRef>([&](const auto *node) {
        TensorShape shape = getShape(node->ref);
        // TODO: verify dim is not out of bounds
        return shape.sizes[node->dim];
      });
}

TensorShape Evaluator::getShape(const ExprNodePtr &node) {
  return getShape(node.get());
}

TensorShape Evaluator::getShape(const ExprNodePtr &node, size_t ordinal) {
  return getShape(node.get(), ordinal);
}

llvm::ArrayRef<TensorShape> Evaluator::getShapes(const ExprNodePtr &node) {
  return getShapes(node.get());
}

TensorShape Evaluator::getShape(const ExprNode *node) {
  auto shapes = getShapes(node);
  if (shapes.size() != 1) {
    throw std::runtime_error(
        "Missing element needed to resolve multlple results.");
  }
  return shapes[0];
}

TensorShape Evaluator::getShape(const ExprNode *node, size_t ordinal) {
  auto shapes = getShapes(node);
  if (shapes.size() < ordinal) {
    throw std::runtime_error("Out of bounds element ordinal");
  }
  return shapes[ordinal];
}

llvm::ArrayRef<TensorShape> Evaluator::getShapes(const ExprNode *node) {
  auto it = shapesCache.find(node);
  if (it == shapesCache.end()) {
    std::tie(it, std::ignore) = shapesCache.emplace(node, computeShapes(node));
  }
  return it->second;
}

TensorShapes Evaluator::computeShapes(const ExprNode *node) {
  TensorShapes shapes =
      llvm::TypeSwitch<const ExprNode *, TensorShapes>(node)
          .Case<ExprNodeCast>([&](const auto *node) {
            TensorShape shape = getShape(node->expr);
            shape.elementType = node->dtype;
            return TensorShapes{shape};
          })
          .Case<ExprNodeConstSigned>([&](const auto *node) {
            return TensorShapes{TensorShape{DataType::six}};
          })
          .Case<ExprNodeConstUnsigned>([&](const auto *node) {
            return TensorShapes{TensorShape{DataType::uix}};
          })
          .Case<ExprNodeConstFloat>([&](const auto *node) {
            return TensorShapes{TensorShape{DataType::fx}};
          })
          .Case<ExprNodeConstTensor>([&](const auto *node) {
            return TensorShapes{node->buffer->shape()};
          })
          .Case<ExprNodeContraction>([&](const auto *node) {
            DataType elementType =
                inferElementType(this, node->comboKind, node->srcs);
            if (isAmbiguousDataType(elementType)) {
              throw std::runtime_error(
                  "'contraction' operand data type is ambiguous, use a cast");
            }
            TensorShape shape{elementType};
            for (const DimNodePtr &dim : node->sinkDims) {
              shape.sizes.push_back(evaluate(dim));
            }
            return TensorShapes{shape};
          })
          .Case<ExprNodeDim>([&](const auto *node) {
            return TensorShapes{TensorShape(DataType::six)};
          })
          .Case<ExprNodeElement>([&](const auto *node) {
            return TensorShapes{getShape(node->expr, node->ordinal)};
          })
          .Case<ExprNodeInput>(
              [&](const auto *node) { return TensorShapes{node->shape}; })
          .Case<ExprNodeIntrinsic>([&](const auto *node) {
            llvm::SmallVector<TensorShape, 8> shapes;
            for (const ExprNodePtr &operand : node->operands) {
              shapes.emplace_back(getShape(operand));
            }
            auto intrinsic = IntrinsicRegistry::Instance()->resolve(node->op);
            if (intrinsic) {
              return intrinsic->getShapes(this, node->operands, shapes);
            }
            return TensorShapes{inferShape(shapes)};
          })
          .Case<ExprNodeTrace>([&](const auto *node) {
            auto shapes = getShapes(node->expr);
            return TensorShapes(shapes.begin(), shapes.end());
          });
  return shapes;
}

} // namespace pmlc::ast
