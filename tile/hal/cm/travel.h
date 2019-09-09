// Copyright 2017-2018 Intel Corporation.

#pragma once

#include <map>
#include <memory>
#include <set>
#include <string>
#include <vector>

#include "tile/base/hal.h"
#include "tile/lang/generate.h"

namespace vertexai {
namespace tile {
namespace hal {
namespace cm {

class TravelVisitor : public sem::Visitor {
 public:
  enum TravelType { CHECK_CM_VECTOR, GET_STRING, GET_GLOBAL_LOAD_EXPRS, GET_GLOBAL_VAR_WITH_OFFSET, GET_INDEX_STRIDE };

  void Visit(const sem::IntConst& node) override;
  void Visit(const sem::FloatConst& node) override;
  void Visit(const sem::LookupLVal& node) override;
  void Visit(const sem::LoadExpr& node) override;
  void Visit(const sem::StoreStmt& node) override;
  void Visit(const sem::SubscriptLVal& node) override;
  void Visit(const sem::DeclareStmt& node) override;
  void Visit(const sem::UnaryExpr& node) override;
  void Visit(const sem::BinaryExpr& node) override;
  void Visit(const sem::CondExpr& node) override;
  void Visit(const sem::SelectExpr& node) override;
  void Visit(const sem::ClampExpr& node) override;
  void Visit(const sem::CastExpr& node) override;
  void Visit(const sem::CallExpr& node) override;
  void Visit(const sem::LimitConst& node) override;
  void Visit(const sem::IndexExpr& node) override;
  void Visit(const sem::Block& node) override;
  void Visit(const sem::IfStmt& node) override;
  void Visit(const sem::ForStmt& node) override;
  void Visit(const sem::WhileStmt& node) override;
  void Visit(const sem::BarrierStmt& node) override;
  void Visit(const sem::ReturnStmt& node) override;
  void Visit(const sem::SpecialStmt& node) override;
  void Visit(const sem::Function& node) override;

  void InitCheckVector() {
    travel = CHECK_CM_VECTOR;
    is_cm_vector = false;
  }

  void InitIndexStride() {
    travel = GET_INDEX_STRIDE;
    index_stride = 0;
  }

  void InitNodeStr() {
    travel = GET_STRING;
    node_str.str("");
  }

  void InitGlobalVarWithOffset() {
    travel = GET_GLOBAL_VAR_WITH_OFFSET;
    global_var_with_offset.str("");
  }

  void InitGlobalLoadExprMap() {
    travel = GET_GLOBAL_LOAD_EXPRS;
    global_load_exprs.clear();
  }

  bool CheckVector() const { return is_cm_vector; }
  int GetIndexStride() const { return index_stride; }
  std::string GetNodeStr() const { return node_str.str(); }
  std::string GetGlobalVarWithOffset() const { return global_var_with_offset.str(); }
  std::map<std::shared_ptr<sem::LoadExpr>, std::string> GetGlobalLoadExprMap() const { return global_load_exprs; }

  std::set<std::string> global_params;
  std::set<std::string> vector_params;
  std::map<std::string, int> index_stride_map;

 private:
  TravelType travel;
  bool is_cm_vector;
  int index_stride;
  std::ostringstream node_str;
  std::ostringstream global_var_with_offset;
  std::map<std::shared_ptr<sem::LoadExpr>, std::string> global_load_exprs;
};

}  // namespace cm
}  // namespace hal
}  // namespace tile
}  // namespace vertexai
