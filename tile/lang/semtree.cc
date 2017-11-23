
#include "tile/lang/semtree.h"

namespace vertexai {
namespace tile {
namespace sem {

void Type::log(el::base::type::ostream_t& os) const { os << to_string(*this); }

std::string to_string(const Type& ty) {
  std::ostringstream os;
  if (ty.region == Type::LOCAL) {
    os << "local ";
  } else if (ty.region == Type::GLOBAL) {
    os << "global ";
  }
  if (ty.base == Type::POINTER_CONST) {
    os << "const ";
  }
  if (ty.base == Type::TVOID) {
    os << "void ";
  }
  if (ty.base == Type::INDEX) {
    os << "index ";
  }
  if (ty.base != Type::TVOID && ty.base != Type::INDEX) {
    os << to_string(ty.dtype);
  }
  if (ty.vec_width > 1) {
    os << 'x' << std::to_string(ty.vec_width);
  }
  if (ty.base == Type::POINTER_MUT || ty.base == Type::POINTER_CONST) {
    os << '*';
  }
  if (ty.array) {
    os << '[' << std::to_string(ty.array) << ']';
  }
  return os.str();
}

void IntConst::Accept(Visitor& v) const { v.Visit(*this); }

void FloatConst::Accept(Visitor& v) const { v.Visit(*this); }

void LookupLVal::Accept(Visitor& v) const { v.Visit(*this); }

void LoadExpr::Accept(Visitor& v) const { v.Visit(*this); }

void StoreStmt::Accept(Visitor& v) const { v.Visit(*this); }

void SubscriptLVal::Accept(Visitor& v) const { v.Visit(*this); }

void DeclareStmt::Accept(Visitor& v) const { v.Visit(*this); }

void UnaryExpr::Accept(Visitor& v) const { v.Visit(*this); }

void BinaryExpr::Accept(Visitor& v) const { v.Visit(*this); }

void CondExpr::Accept(Visitor& v) const { v.Visit(*this); }

void SelectExpr::Accept(Visitor& v) const { v.Visit(*this); }

void ClampExpr::Accept(Visitor& v) const { v.Visit(*this); }

void CastExpr::Accept(Visitor& v) const { v.Visit(*this); }

void CallExpr::Accept(Visitor& v) const { v.Visit(*this); }

void LimitConst::Accept(Visitor& v) const { v.Visit(*this); }

void IndexExpr::Accept(Visitor& v) const { v.Visit(*this); }

void Block::merge(std::shared_ptr<Block> other) {
  statements.insert(statements.end(), other->statements.begin(), other->statements.end());
}

void Block::append(StmtPtr p) {
  if (p) {
    if (p->isBlock()) {
      merge(std::static_pointer_cast<Block>(p));
    } else {
      push_back(p);
    }
  }
}

void Block::Accept(Visitor& v) const { v.Visit(*this); }

IfStmt::IfStmt(ExprPtr c, StmtPtr t, StmtPtr f) : cond(c), iftrue(t), iffalse(f) {
  if (iftrue && !iftrue->isBlock()) {
    iftrue = std::make_shared<Block>(std::vector<StmtPtr>{iftrue});
  }
  if (iffalse && !iffalse->isBlock()) {
    iffalse = std::make_shared<Block>(std::vector<StmtPtr>{iffalse});
  }
}

void IfStmt::Accept(Visitor& v) const { v.Visit(*this); }

ForStmt::ForStmt(const std::string v, uint64_t n, uint64_t s, StmtPtr i) : var(v), num(n), step(s), inner(i) {
  if (!inner->isBlock()) {
    inner = std::make_shared<Block>(std::vector<StmtPtr>{i});
  }
}

void ForStmt::Accept(Visitor& v) const { v.Visit(*this); }

WhileStmt::WhileStmt(ExprPtr c, StmtPtr i) : cond(c), inner(i) {
  if (!inner->isBlock()) {
    inner = std::make_shared<Block>(std::vector<StmtPtr>{i});
  }
}

void WhileStmt::Accept(Visitor& v) const { v.Visit(*this); }

void BarrierStmt::Accept(Visitor& v) const { v.Visit(*this); }

void ReturnStmt::Accept(Visitor& v) const { v.Visit(*this); }

Function::Function(const std::string n, const Type& r, const params_t& p, StmtPtr b)
    : name(n), ret(r), params(p), body(b) {
  if (!body->isBlock()) {
    body = std::make_shared<Block>(std::vector<StmtPtr>{body});
  }
}

void Function::Accept(Visitor& v) const { v.Visit(*this); }

}  // namespace sem
}  // namespace tile
}  // namespace vertexai
