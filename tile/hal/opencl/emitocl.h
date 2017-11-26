// Copyright 2017, Vertex.AI.

#pragma once

#include <sstream>
#include <string>

#include "tile/lang/emitc.h"
#include "tile/lang/scope.h"

namespace vertexai {
namespace tile {
namespace hal {
namespace opencl {

class Emit final : public sem::EmitC {
 public:
  Emit(bool cl_khr_fp16, bool cl_khr_fp64) : cl_khr_fp16_{cl_khr_fp16}, cl_khr_fp64_{cl_khr_fp64}, scope_{nullptr} {}

  void operator()(const sem::IntConst& v) final { EmitC::operator()(v); }
  void operator()(const sem::FloatConst& v) final { EmitC::operator()(v); }
  void operator()(const sem::LookupLVal& v) final { EmitC::operator()(v); }
  void operator()(const sem::SubscriptLVal& v) final { EmitC::operator()(v); }
  void operator()(const sem::UnaryExpr& v) final { EmitC::operator()(v); }
  void operator()(const sem::LimitConst& v) final { EmitC::operator()(v); }
  void operator()(const sem::IfStmt& v) final { EmitC::operator()(v); }
  void operator()(const sem::WhileStmt& v) final { EmitC::operator()(v); }
  void operator()(const sem::ReturnStmt& v) final { EmitC::operator()(v); }

  void operator()(const sem::LoadExpr&) final;
  void operator()(const sem::StoreStmt&) final;
  void operator()(const sem::DeclareStmt&) final;
  void operator()(const sem::BinaryExpr&) final;
  void operator()(const sem::CondExpr& n) final;
  void operator()(const sem::SelectExpr& n) final;
  void operator()(const sem::ClampExpr& n) final;
  void operator()(const sem::CastExpr&) final;
  void operator()(const sem::CallExpr&) final;
  void operator()(const sem::IndexExpr&) final;
  void operator()(const sem::Block&) final;
  void operator()(const sem::ForStmt&) final;
  void operator()(const sem::BarrierStmt&) final;
  void operator()(const sem::Function&) final;

 private:
  void CheckValidType(const sem::Type& ty);
  sem::Type TypeOf(const sem::ExprPtr& expr);
  sem::Type TypeOf(const sem::LValPtr& lvalue);
  void EmitWithTypeConversion(const sem::Type& from, const sem::Type& to, const sem::ExprPtr& expr,
                              bool force_conversion = false);
  void EmitWithWidthConversion(const sem::Type& from, const sem::Type& to, const sem::ExprPtr& expr,
                               bool force_conversion = false);
  void emitType(const sem::Type& t) final;

  bool cl_khr_fp16_;
  bool cl_khr_fp64_;
  lang::Scope<sem::Type>* scope_;
};

}  // namespace opencl
}  // namespace hal
}  // namespace tile
}  // namespace vertexai
