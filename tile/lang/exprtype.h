// Copyright 2017-2018 Intel Corporation.

#pragma once

#include <vector>

#include "tile/lang/scope.h"
#include "tile/lang/semtree.h"

namespace vertexai {
namespace tile {
namespace lang {

// Compute the lowest common type for a set of types.
sem::Type Promote(const std::vector<sem::Type>& types);

// Analyzes expression types.
class ExprType final : public sem::Visitor {
 public:
  static sem::Type TypeOf(const lang::Scope<sem::Type>* scope, bool enable_fp16, const sem::ExprPtr& expr);

  static sem::Type TypeOf(const lang::Scope<sem::Type>* scope, bool enable_fp16, const sem::LValPtr& lvalue);

  void Visit(const sem::IntConst&) final;
  void Visit(const sem::FloatConst&) final;
  void Visit(const sem::LookupLVal&) final;
  void Visit(const sem::LoadExpr&) final;
  void Visit(const sem::StoreStmt&) final;
  void Visit(const sem::SubscriptLVal&) final;
  void Visit(const sem::DeclareStmt&) final;
  void Visit(const sem::UnaryExpr&) final;
  void Visit(const sem::BinaryExpr&) final;
  void Visit(const sem::CondExpr&) final;
  void Visit(const sem::SelectExpr&) final;
  void Visit(const sem::ClampExpr&) final;
  void Visit(const sem::CastExpr&) final;
  void Visit(const sem::CallExpr&) final;
  void Visit(const sem::LimitConst&) final;
  void Visit(const sem::IndexExpr&) final;
  void Visit(const sem::Block&) final;
  void Visit(const sem::IfStmt&) final;
  void Visit(const sem::ForStmt&) final;
  void Visit(const sem::WhileStmt&) final;
  void Visit(const sem::BarrierStmt&) final;
  void Visit(const sem::ReturnStmt&) final;
  void Visit(const sem::Function&) final;

 private:
  ExprType(const lang::Scope<sem::Type>* scope, bool enable_fp16);

  sem::Type TypeOf(const sem::ExprPtr& expr);

  void AdjustLogicOpResult();

  const lang::Scope<sem::Type>* scope_;
  bool enable_fp16_;
  sem::Type ty_;
};

}  // namespace lang
}  // namespace tile
}  // namespace vertexai
