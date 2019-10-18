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

class AstPass : public AstVisitor<ExprPtr> {
 public:
  ExprPtr Visit(const CallExpr& expr) override { return GenericVisit(expr); }
  ExprPtr Visit(const ContractionExpr& expr) override { return GenericVisit(expr); }
  ExprPtr Visit(const DimExprExpr& expr) override { return GenericVisit(expr); }
  ExprPtr Visit(const FloatConst& expr) override { return GenericVisit(expr); }
  ExprPtr Visit(const IntConst& expr) override { return GenericVisit(expr); }
  ExprPtr Visit(const ParamExpr& expr) override { return GenericVisit(expr); }
  ExprPtr Visit(const GradOverrideExpr& expr) override { return GenericVisit(expr); }

 protected:
  template <typename T>
  ExprPtr GenericVisit(const T& expr) {
    return ExprPtr{std::const_pointer_cast<Expr>(expr.as_ptr())};
  }
};

std::vector<ExprPtr> FlattenAst(const ProgramMutations& mutations);

ProgramMutations RunAstPass(const ProgramMutations& mutations, AstPass* pass);

}  // namespace ast
}  // namespace lang
}  // namespace tile
}  // namespace vertexai
