// Copyright 2017-2018 Intel Corporation.

#pragma once

#include <sstream>
#include <string>

#include "tile/lang/emitc.h"
#include "tile/lang/scope.h"

namespace vertexai {
namespace tile {
namespace hal {
namespace opencl {

class Emit : public lang::EmitC {
 public:
  explicit Emit(bool cl_khr_fp16, bool cl_khr_fp64)
      : cl_khr_fp16_{cl_khr_fp16}, cl_khr_fp64_{cl_khr_fp64}, scope_{nullptr} {}

  void Visit(const sem::LoadExpr&) final;
  void Visit(const sem::StoreStmt&) final;
  void Visit(const sem::DeclareStmt&) final;
  void Visit(const sem::BinaryExpr&) final;
  void Visit(const sem::CondExpr& n) final;
  void Visit(const sem::SelectExpr& n) final;
  void Visit(const sem::ClampExpr& n) final;
  void Visit(const sem::CastExpr&) final;
  void Visit(const sem::CallExpr&) final;
  void Visit(const sem::IndexExpr&) final;
  void Visit(const sem::Block&) final;
  void Visit(const sem::ForStmt&) final;
  void Visit(const sem::BarrierStmt&) final;
  void Visit(const sem::Function&) final;

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
