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

}  // namespace lang
}  // namespace tile
}  // namespace vertexai
