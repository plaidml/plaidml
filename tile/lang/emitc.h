#pragma once

#include <sstream>
#include <string>

#include "tile/lang/semtree.h"

namespace vertexai {
namespace tile {
namespace sem {

class EmitC : public boost::static_visitor<> {
 public:
  virtual void operator()(const sem::IntConst&);
  virtual void operator()(const sem::FloatConst&);
  virtual void operator()(const sem::LookupLVal&);
  virtual void operator()(const sem::LoadExpr&);
  virtual void operator()(const sem::StoreStmt&);
  virtual void operator()(const sem::SubscriptLVal&);
  virtual void operator()(const sem::DeclareStmt&);
  virtual void operator()(const sem::UnaryExpr&);
  virtual void operator()(const sem::BinaryExpr&);
  virtual void operator()(const sem::CondExpr&);
  virtual void operator()(const sem::SelectExpr&);
  virtual void operator()(const sem::ClampExpr&);
  virtual void operator()(const sem::CastExpr&);
  virtual void operator()(const sem::CallExpr&);
  virtual void operator()(const sem::LimitConst&);
  virtual void operator()(const sem::IndexExpr&);
  virtual void operator()(const sem::Block&);
  virtual void operator()(const sem::IfStmt&);
  virtual void operator()(const sem::ForStmt&);
  virtual void operator()(const sem::WhileStmt&);
  virtual void operator()(const sem::BarrierStmt&);
  virtual void operator()(const sem::ReturnStmt&);
  virtual void operator()(const sem::Function&);
  std::string str() const { return result_.str(); }

 protected:
  void emit(const std::string& s) { result_ << s; }
  virtual void emitType(const sem::Type& t);
  void emitTab() { result_ << std::string(indent_ << 1, ' '); }
  std::ostringstream result_;
  size_t indent_ = 0;
};

class Print final : public EmitC {
 public:
  void operator()(const sem::CallExpr& n) final;
  void operator()(const sem::IndexExpr& n) final;
  void operator()(const sem::BarrierStmt& n) final;
  void operator()(const sem::Function& f) final;
};

inline std::string to_string(const sem::Function& f) {
  Print p{};
  p(f);
  return p.str();
}

}  // namespace sem
}  // namespace tile
}  // namespace vertexai
