
#pragma once

#include <memory>
#include <string>

#include "tile/lang/semtree.h"

namespace vertexai {
namespace tile {
namespace sem {
namespace builder {

// A DSL for building semantic trees

class LValueHolder {
 public:
  explicit LValueHolder(LValPtr v) : v_(v) {}
  operator ExprPtr() const { return std::make_shared<LoadExpr>(v_); }
  std::shared_ptr<StoreStmt> operator=(const LValueHolder& rhs) const {
    return std::make_shared<StoreStmt>(v_, std::make_shared<LoadExpr>(rhs.v_));
  }
  std::shared_ptr<StoreStmt> operator=(int x) const {
    return std::make_shared<StoreStmt>(v_, std::make_shared<IntConst>(x));
  }
  std::shared_ptr<StoreStmt> operator=(ExprPtr rhs) const { return std::make_shared<StoreStmt>(v_, rhs); }
  LValueHolder operator[](ExprPtr offset) const { return LValueHolder(std::make_shared<SubscriptLVal>(v_, offset)); }
  ExprPtr operator()(ExprPtr val) const {
    return std::make_shared<CallExpr>(std::make_shared<LoadExpr>(v_), std::vector<ExprPtr>{val});
  }
  ExprPtr operator()(ExprPtr v1, ExprPtr v2) const {
    return std::make_shared<CallExpr>(std::make_shared<LoadExpr>(v_), std::vector<ExprPtr>{v1, v2});
  }
  ExprPtr operator()(ExprPtr v1, ExprPtr v2, ExprPtr v3) const {
    return std::make_shared<CallExpr>(std::make_shared<LoadExpr>(v_), std::vector<ExprPtr>{v1, v2, v3});
  }
  ExprPtr operator()(int val) const {
    return std::make_shared<CallExpr>(std::make_shared<LoadExpr>(v_),
                                      std::vector<ExprPtr>({std::make_shared<IntConst>(val)}));
  }

 private:
  LValPtr v_;
};

inline LValueHolder _(const std::string& name) { return LValueHolder(std::make_shared<LookupLVal>(name)); }

inline std::shared_ptr<LimitConst> _LimitConst(const LimitConst::Which& which, const lang::DataType& type) {
  return std::make_shared<LimitConst>(which, type);
}

inline std::shared_ptr<IndexExpr> _Index(const IndexExpr::Type& type, size_t dim) {
  return std::make_shared<IndexExpr>(type, dim);
}

inline std::shared_ptr<Block> _Block(std::initializer_list<StmtPtr> inner) {
  return std::make_shared<Block>(std::vector<StmtPtr>(inner));
}

inline std::shared_ptr<IfStmt> _If(ExprPtr cond, StmtPtr iftrue) {
  return std::make_shared<IfStmt>(cond, iftrue, StmtPtr());
}
inline std::shared_ptr<IfStmt> _If(ExprPtr cond, StmtPtr iftrue, StmtPtr iffalse) {
  return std::make_shared<IfStmt>(cond, iftrue, iffalse);
}

inline std::shared_ptr<WhileStmt> _While(ExprPtr cond, StmtPtr inner) {
  return std::make_shared<WhileStmt>(cond, inner);
}
inline std::shared_ptr<ForStmt> _For(const std::string& var, uint64_t n, uint64_t s, StmtPtr inner) {
  return std::make_shared<ForStmt>(var, n, s, inner);
}

inline std::shared_ptr<BreakStmt> _Break() { return std::make_shared<BreakStmt>(); }
inline std::shared_ptr<ContinueStmt> _Continue() { return std::make_shared<ContinueStmt>(); }
inline std::shared_ptr<BarrierStmt> _Barrier() { return std::make_shared<BarrierStmt>(); }
inline std::shared_ptr<ReturnStmt> _Return(ExprPtr value = ExprPtr()) { return std::make_shared<ReturnStmt>(value); }

inline std::shared_ptr<Function> _Function(const std::string& name, const Type& ret,
                                           std::initializer_list<Function::param_t> params,
                                           std::initializer_list<StmtPtr> body) {
  return std::make_shared<Function>(name, ret, std::vector<Function::param_t>(params),
                                    std::make_shared<Block>(std::vector<StmtPtr>(body)));
}

inline std::shared_ptr<IntConst> _Const(int x) { return std::make_shared<IntConst>(x); }
inline std::shared_ptr<IntConst> _Const(int64_t x) { return std::make_shared<IntConst>(x); }
inline std::shared_ptr<IntConst> _Const(uint64_t x) { return std::make_shared<IntConst>(x); }
inline std::shared_ptr<FloatConst> _Const(double x) { return std::make_shared<FloatConst>(x); }
inline std::shared_ptr<DeclareStmt> _Declare(const Type& type, const std::string& name, ExprPtr init) {
  return std::make_shared<DeclareStmt>(type, name, init);
}
inline std::shared_ptr<DeclareStmt> _DeclareConst(const Type& type, const std::string& name, int init) {
  return _Declare(type, name, _Const(init));
}
inline std::shared_ptr<CastExpr> _Cast(const Type& type, ExprPtr init) {
  return std::make_shared<CastExpr>(type, init);
}

inline std::shared_ptr<CondExpr> _Cond(ExprPtr cond, ExprPtr tcase, ExprPtr fcase) {
  return std::make_shared<CondExpr>(cond, tcase, fcase);
}
inline std::shared_ptr<SelectExpr> _Select(ExprPtr cond, ExprPtr tcase, ExprPtr fcase) {
  return std::make_shared<SelectExpr>(cond, tcase, fcase);
}
inline std::shared_ptr<ClampExpr> _Clamp(ExprPtr val, ExprPtr min, ExprPtr max) {
  return std::make_shared<ClampExpr>(val, min, max);
}

}  // namespace builder
}  // namespace sem
}  // namespace tile
}  // namespace vertexai

/* Import the ops into the std namespace to use ADL on std::shared_ptr */

// Define some unary ops
#define DEF_UNARY_OP(op)                                                                                   \
  inline std::shared_ptr<vertexai::tile::sem::UnaryExpr> operator op(vertexai::tile::sem::ExprPtr inner) { \
    return std::make_shared<vertexai::tile::sem::UnaryExpr>(#op, inner);                                   \
  }

// Define some binary ops
#define DEF_BINARY_OP(op)                                                                                          \
  inline std::shared_ptr<vertexai::tile::sem::BinaryExpr> operator op(vertexai::tile::sem::ExprPtr lhs,            \
                                                                      vertexai::tile::sem::ExprPtr rhs) {          \
    return std::make_shared<vertexai::tile::sem::BinaryExpr>(#op, lhs, rhs);                                       \
  }                                                                                                                \
  inline std::shared_ptr<vertexai::tile::sem::BinaryExpr> operator op(vertexai::tile::sem::ExprPtr lhs, int rhs) { \
    return std::make_shared<vertexai::tile::sem::BinaryExpr>(#op, lhs, vertexai::tile::sem::builder::_Const(rhs)); \
  }                                                                                                                \
  inline std::shared_ptr<vertexai::tile::sem::BinaryExpr> operator op(int lhs, vertexai::tile::sem::ExprPtr rhs) { \
    return std::make_shared<vertexai::tile::sem::BinaryExpr>(#op, vertexai::tile::sem::builder::_Const(lhs), rhs); \
  }

namespace std {

DEF_UNARY_OP(!)
DEF_UNARY_OP(-)
DEF_UNARY_OP(~)

DEF_BINARY_OP(*)
DEF_BINARY_OP(%)
DEF_BINARY_OP(/)
DEF_BINARY_OP(+)
DEF_BINARY_OP(-)
DEF_BINARY_OP(==)
DEF_BINARY_OP(!=)
DEF_BINARY_OP(>)  // NOLINT
DEF_BINARY_OP(<)  // NOLINT
DEF_BINARY_OP(>=)
DEF_BINARY_OP(<=)
DEF_BINARY_OP(&)
DEF_BINARY_OP(|)
DEF_BINARY_OP (^)  // NOLINT
DEF_BINARY_OP(>>)
DEF_BINARY_OP(<<)
}  // End namespace std

using namespace std;  // NOLINT
