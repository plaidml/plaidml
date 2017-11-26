// Copyright 2017, Vertex.AI.

#pragma once

#include <vector>

#include "tile/lang/scope.h"
#include "tile/lang/semtree.h"

namespace vertexai {
namespace tile {
namespace hal {
namespace opencl {

// TODO: Consider whether this logic belongs in a more general
// location, since it doesn't actually make use of OpenCL types or the
// OpenCL type hierarchy, with the exception of whether the system
// supports half-width floats.

// Compute the lowest common type for a set of types.
sem::Type Promote(const std::vector<sem::Type>& types);

// Analyzes expression types.
class ExprType final : public boost::static_visitor<sem::Type> {
 public:
  static sem::Type TypeOf(const lang::Scope<sem::Type>* scope, bool cl_khr_fp16, const sem::ExprPtr& expr);

  static sem::Type TypeOf(const lang::Scope<sem::Type>* scope, bool cl_khr_fp16, const sem::LValPtr& lvalue);

  static sem::Type AdjustLogicOpResult(sem::Type ty);

  ExprType(const lang::Scope<sem::Type>* scope, bool cl_khr_fp16) : scope_{scope}, cl_khr_fp16_{cl_khr_fp16} {}

  sem::Type operator()(const sem::IntConst&);
  sem::Type operator()(const sem::FloatConst&);
  sem::Type operator()(const sem::LookupLVal&);
  sem::Type operator()(const sem::LoadExpr&);
  sem::Type operator()(const sem::SubscriptLVal&);
  sem::Type operator()(const sem::UnaryExpr&);
  sem::Type operator()(const sem::BinaryExpr&);
  sem::Type operator()(const sem::CondExpr&);
  sem::Type operator()(const sem::SelectExpr&);
  sem::Type operator()(const sem::ClampExpr&);
  sem::Type operator()(const sem::CastExpr&);
  sem::Type operator()(const sem::CallExpr&);
  sem::Type operator()(const sem::LimitConst&);
  sem::Type operator()(const sem::IndexExpr&);

 private:
  const lang::Scope<sem::Type>* scope_;
  const bool cl_khr_fp16_;
};

}  // namespace opencl
}  // namespace hal
}  // namespace tile
}  // namespace vertexai
