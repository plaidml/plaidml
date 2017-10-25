#pragma once

#include <sstream>
#include <string>

#include "tile/lang/semtree.h"

namespace vertexai {
namespace tile {
namespace lang {

class EmitC : public sem::Visitor {
 public:
  void Visit(const sem::IntConst &) override;
  void Visit(const sem::FloatConst &) override;
  void Visit(const sem::LookupLVal &) override;
  void Visit(const sem::LoadExpr &) override;
  void Visit(const sem::StoreStmt &) override;
  void Visit(const sem::SubscriptLVal &) override;
  void Visit(const sem::DeclareStmt &) override;
  void Visit(const sem::UnaryExpr &) override;
  void Visit(const sem::BinaryExpr &) override;
  void Visit(const sem::CondExpr &) override;
  void Visit(const sem::SelectExpr &) override;
  void Visit(const sem::ClampExpr &) override;
  void Visit(const sem::CastExpr &) override;
  void Visit(const sem::CallExpr &) override;
  void Visit(const sem::LimitConst &) override;
  void Visit(const sem::IndexExpr &) override;
  void Visit(const sem::Block &) override;
  void Visit(const sem::IfStmt &) override;
  void Visit(const sem::ForStmt &) override;
  void Visit(const sem::WhileStmt &) override;
  void Visit(const sem::BreakStmt &) override;
  void Visit(const sem::ContinueStmt &) override;
  void Visit(const sem::BarrierStmt &) override;
  void Visit(const sem::ReturnStmt &) override;
  void Visit(const sem::Function &) override;
  std::string str() const { return result_.str(); }

 protected:
  void emit(const std::string &s) { result_ << s; }
  virtual void emitType(const sem::Type &t);
  void emitTab() { result_ << std::string(indent_ << 1, ' '); }
  std::ostringstream result_;
  size_t indent_ = 0;
};

class EmitDebug : public EmitC {
 public:
  void Visit(const sem::CallExpr &n) final {
    n.func->Accept(*this);
    emit("(");
    for (size_t i = 0; i < n.vals.size(); i++) {
      n.vals[i]->Accept(*this);
      if (i != n.vals.size() - 1) {
        emit(", ");
      }
    }
    emit(")");
  }

  void Visit(const sem::IndexExpr &n) final {
    switch (n.type) {
      case sem::IndexExpr::GLOBAL:
        emit("get_global_id(" + std::to_string(n.dim) + ")");
        break;
      case sem::IndexExpr::GROUP:
        emit("get_group_id(" + std::to_string(n.dim) + ")");
        break;
      case sem::IndexExpr::LOCAL:
        emit("get_local_id(" + std::to_string(n.dim) + ")");
        break;
      default:
        throw std::runtime_error("Invalid IndexExpr type");
    }
  }

  void Visit(const sem::BarrierStmt &n) {
    emitTab();
    emit("barrier();\n");
  }

  void Visit(const sem::Function &n) final {
    emit("kernel ");
    EmitC::Visit(n);
  }
};

}  // namespace lang
}  // namespace tile
}  // namespace vertexai
