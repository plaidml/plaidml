
#pragma once

#include <sstream>
#include <string>

#include "tile/lang/semtree.h"

namespace vertexai {
namespace tile {
namespace sem {

class Print : public Visitor {
 public:
  Print() {}
  explicit Print(const Node& n) { n.Accept(*this); }
  void Visit(const IntConst&) override;
  void Visit(const FloatConst&) override;
  void Visit(const LookupLVal&) override;
  void Visit(const LoadExpr&) override;
  void Visit(const StoreStmt&) override;
  void Visit(const SubscriptLVal&) override;
  void Visit(const DeclareStmt&) override;
  void Visit(const UnaryExpr&) override;
  void Visit(const BinaryExpr&) override;
  void Visit(const CondExpr&) override;
  void Visit(const SelectExpr&) override;
  void Visit(const ClampExpr&) override;
  void Visit(const CastExpr&) override;
  void Visit(const CallExpr&) override;
  void Visit(const LimitConst&) override;
  void Visit(const IndexExpr&) override;
  void Visit(const Block&) override;
  void Visit(const IfStmt&) override;
  void Visit(const ForStmt&) override;
  void Visit(const WhileStmt&) override;
  void Visit(const BarrierStmt&) override;
  void Visit(const ReturnStmt&) override;
  void Visit(const Function&) override;
  std::string str() const { return result_.str(); }

 protected:
  void emit(const std::string& s) { result_ << s; }
  virtual void emitType(const Type& t);
  void emitTab() { result_ << std::string(indent_ << 1, ' '); }
  std::ostringstream result_;
  size_t indent_ = 0;
};

}  // namespace sem
}  // namespace tile
}  // namespace vertexai
