
#include "tile/lang/semtree.h"

namespace vertexai {
namespace tile {
namespace sem {

void IntConst::Accept(Visitor &v) const { v.Visit(*this); }

void FloatConst::Accept(Visitor &v) const { v.Visit(*this); }

void LookupLVal::Accept(Visitor &v) const { v.Visit(*this); }

void LoadExpr::Accept(Visitor &v) const { v.Visit(*this); }

void StoreStmt::Accept(Visitor &v) const { v.Visit(*this); }

void SubscriptLVal::Accept(Visitor &v) const { v.Visit(*this); }

void DeclareStmt::Accept(Visitor &v) const { v.Visit(*this); }

void UnaryExpr::Accept(Visitor &v) const { v.Visit(*this); }

void BinaryExpr::Accept(Visitor &v) const { v.Visit(*this); }

void CondExpr::Accept(Visitor &v) const { v.Visit(*this); }

void SelectExpr::Accept(Visitor &v) const { v.Visit(*this); }

void ClampExpr::Accept(Visitor &v) const { v.Visit(*this); }

void CastExpr::Accept(Visitor &v) const { v.Visit(*this); }

void CallExpr::Accept(Visitor &v) const { v.Visit(*this); }

void LimitConst::Accept(Visitor &v) const { v.Visit(*this); }

void IndexExpr::Accept(Visitor &v) const { v.Visit(*this); }

void Block::merge(std::shared_ptr<Block> other) {
  statements.insert(statements.end(), other->statements.begin(), other->statements.end());
}

void Block::append(StmtPtr p) {
  if (p->isBlock()) {
    merge(std::static_pointer_cast<Block>(p));
  } else {
    push_back(p);
  }
}

void Block::Accept(Visitor &v) const { v.Visit(*this); }

IfStmt::IfStmt(ExprPtr c, StmtPtr t, StmtPtr f) : cond(c), iftrue(t), iffalse(f) {
  if (!iftrue->isBlock()) {
    iftrue = std::make_shared<Block>(std::vector<StmtPtr>{iftrue});
  }
  if (iffalse && !iffalse->isBlock()) {
    iffalse = std::make_shared<Block>(std::vector<StmtPtr>{iffalse});
  }
}

void IfStmt::Accept(Visitor &v) const { v.Visit(*this); }

ForStmt::ForStmt(const std::string v, uint64_t n, uint64_t s, StmtPtr i) : var(v), num(n), step(s), inner(i) {
  if (!inner->isBlock()) {
    inner = std::make_shared<Block>(std::vector<StmtPtr>{i});
  }
}

void ForStmt::Accept(Visitor &v) const { v.Visit(*this); }

WhileStmt::WhileStmt(ExprPtr c, StmtPtr i) : cond(c), inner(i) {
  if (!inner->isBlock()) {
    inner = std::make_shared<Block>(std::vector<StmtPtr>{i});
  }
}

void WhileStmt::Accept(Visitor &v) const { v.Visit(*this); }

void BreakStmt::Accept(Visitor &v) const { v.Visit(*this); }

void ContinueStmt::Accept(Visitor &v) const { v.Visit(*this); }

void BarrierStmt::Accept(Visitor &v) const { v.Visit(*this); }

void ReturnStmt::Accept(Visitor &v) const { v.Visit(*this); }

Function::Function(const std::string n, const Type &r, const params_t &p, StmtPtr b)
    : name(n), ret(r), params(p), body(b) {
  if (!body->isBlock()) {
    body = std::make_shared<Block>(std::vector<StmtPtr>{body});
  }
}

void Function::Accept(Visitor &v) const { v.Visit(*this); }

}  // namespace sem
}  // namespace tile
}  // namespace vertexai
