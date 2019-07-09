// Copyright 2019 Intel Corporation.

#pragma once

#include <stack>
#include <unordered_set>
#include <utility>
#include <vector>

#include "tile/lang/ast/ast.h"

namespace vertexai {
namespace tile {
namespace lang {
namespace ast {

class AstTraversal : public AstVisitor {
 public:
  explicit AstTraversal(const std::vector<ExprPtr>& exprs);
  const std::vector<ExprPtr>& flat() const { return flat_; }

 private:
  void Visit(const CallExpr& expr);
  void Visit(const ConstraintExpr& expr);
  void Visit(const ContractionExpr& expr);
  void Visit(const FloatConst& expr);
  void Visit(const IntConst& expr);
  void Visit(const ParamExpr& expr);
  void Visit(const TensorSpecExpr& expr);
  void Visit(const DimExprExpr& expr);

 private:
  void Push(const ExprPtr& expr);

 private:
  std::stack<std::pair<ExprPtr, bool>> stack_;
  std::vector<ExprPtr> flat_;
  std::unordered_set<const Expr*> seen_;
};

}  // namespace ast
}  // namespace lang
}  // namespace tile
}  // namespace vertexai
