
#pragma once

#include <memory>
#include <string>
#include <vector>

#include "tile/lang/semtree.h"

namespace vertexai {
namespace tile {
namespace sem {
namespace builder {

template <typename T>
inline std::shared_ptr<IntConst> _Const(T x) {
  return std::make_shared<IntConst>(x);
}

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

inline std::shared_ptr<LimitConst> _LimitConst(const LimitConst::Which& which, const DataType& type) {
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

inline std::shared_ptr<BarrierStmt> _Barrier(bool subgroup = false) { return std::make_shared<BarrierStmt>(subgroup); }

inline std::shared_ptr<ReturnStmt> _Return(ExprPtr value = ExprPtr()) { return std::make_shared<ReturnStmt>(value); }

inline std::shared_ptr<SpecialStmt> _Special(const std::string& name, std::initializer_list<ExprPtr> args) {
  return std::make_shared<SpecialStmt>(name, std::vector<ExprPtr>(args));
}

inline std::shared_ptr<Function> _Function(const std::string& name, const Type& ret,
                                           std::initializer_list<Function::param_t> params,
                                           std::initializer_list<StmtPtr> body) {
  return std::make_shared<Function>(name, ret, std::vector<Function::param_t>(params),
                                    std::make_shared<Block>(std::vector<StmtPtr>(body)));
}

inline std::shared_ptr<FloatConst> _Const(double x) { return std::make_shared<FloatConst>(x); }

inline std::shared_ptr<DeclareStmt> _Declare(const Type& type, const std::string& name, ExprPtr init) {
  return std::make_shared<DeclareStmt>(type, name, init);
}

inline LValueHolder _Declare(std::shared_ptr<Block> block, const Type& type, const std::string& name, ExprPtr init) {
  block->append(_Declare(type, name, init));
  return _(name);
}

template <typename T>
inline std::shared_ptr<DeclareStmt> _DeclareConst(const Type& type, const std::string& name, T init) {
  return _Declare(type, name, _Const(init));
}

inline std::shared_ptr<CastExpr> _Cast(const Type& type, ExprPtr init) {
  return std::make_shared<CastExpr>(type, init);
}

inline std::shared_ptr<CondExpr> _Cond(ExprPtr cond, ExprPtr tcase, ExprPtr fcase) {
  return std::make_shared<CondExpr>(cond, tcase, fcase);
}

inline std::shared_ptr<CondExpr> _Cond(ExprPtr cond, ExprPtr tcase, ExprPtr fcase, Type type) {
  return std::make_shared<CondExpr>(cond, tcase, fcase, type);
}

inline std::shared_ptr<SelectExpr> _Select(ExprPtr cond, ExprPtr tcase, ExprPtr fcase) {
  return std::make_shared<SelectExpr>(cond, tcase, fcase);
}

inline std::shared_ptr<SelectExpr> _Select(ExprPtr cond, ExprPtr tcase, ExprPtr fcase, Type type) {
  return std::make_shared<SelectExpr>(cond, tcase, fcase, type);
}

inline std::shared_ptr<ClampExpr> _Clamp(ExprPtr val, ExprPtr min, ExprPtr max) {
  return std::make_shared<ClampExpr>(val, min, max);
}

inline std::shared_ptr<BinaryExpr> _LogicalAnd(ExprPtr lhs, ExprPtr rhs) {
  return std::make_shared<BinaryExpr>("&&", lhs, rhs);
}

inline std::shared_ptr<BinaryExpr> _LogicalOr(ExprPtr lhs, ExprPtr rhs) {
  return std::make_shared<BinaryExpr>("||", lhs, rhs);
}

inline ExprPtr _MaybeSelect(ExprPtr cond, ExprPtr tcase, ExprPtr fcase, Type type) {
  if (cond) {
    return _Select(cond, tcase, fcase, type);
  }
  return tcase;
}

inline ExprPtr _MaybeCond(ExprPtr cond, ExprPtr tcase, ExprPtr fcase, Type type) {
  if (cond) {
    return _Cond(cond, tcase, fcase, type);
  }
  return tcase;
}

inline ExprPtr _MaybeLogicalAnd(ExprPtr lhs, ExprPtr rhs) {
  if (lhs.get() && rhs.get()) {
    return _LogicalAnd(lhs, rhs);
  }
  return lhs.get() ? lhs : rhs;
}

inline ExprPtr _MaybeLogicalOr(ExprPtr lhs, ExprPtr rhs) {
  if (lhs.get() && rhs.get()) {
    return _LogicalOr(lhs, rhs);
  }
  return lhs.get() ? lhs : rhs;
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
#define DEF_BINARY_OP(op)                                                                                             \
  inline std::shared_ptr<vertexai::tile::sem::BinaryExpr> operator op(vertexai::tile::sem::ExprPtr lhs,               \
                                                                      vertexai::tile::sem::ExprPtr rhs) {             \
    return std::make_shared<vertexai::tile::sem::BinaryExpr>(#op, lhs, rhs);                                          \
  }                                                                                                                   \
  template <class T, typename std::enable_if<std::is_integral<T>::value>::type* = nullptr>                            \
  inline std::shared_ptr<vertexai::tile::sem::BinaryExpr> operator op(vertexai::tile::sem::ExprPtr lhs, T rhs) {      \
    auto rhs_const = vertexai::tile::sem::builder::_Const(rhs);                                                       \
    return std::make_shared<vertexai::tile::sem::BinaryExpr>(#op, lhs, rhs_const);                                    \
  }                                                                                                                   \
  template <class T, typename std::enable_if<std::is_integral<T>::value>::type* = nullptr>                            \
  inline std::shared_ptr<vertexai::tile::sem::BinaryExpr> operator op(T lhs, vertexai::tile::sem::ExprPtr rhs) {      \
    auto lhs_const = vertexai::tile::sem::builder::_Const(lhs);                                                       \
    return std::make_shared<vertexai::tile::sem::BinaryExpr>(#op, lhs_const, rhs);                                    \
  }                                                                                                                   \
  inline std::shared_ptr<vertexai::tile::sem::BinaryExpr> operator op(vertexai::tile::sem::ExprPtr lhs, double rhs) { \
    auto rhs_const = vertexai::tile::sem::builder::_Const(rhs);                                                       \
    return std::make_shared<vertexai::tile::sem::BinaryExpr>(#op, lhs, rhs_const);                                    \
  }                                                                                                                   \
  inline std::shared_ptr<vertexai::tile::sem::BinaryExpr> operator op(double lhs, vertexai::tile::sem::ExprPtr rhs) { \
    auto lhs_const = vertexai::tile::sem::builder::_Const(lhs);                                                       \
    return std::make_shared<vertexai::tile::sem::BinaryExpr>(#op, lhs_const, rhs);                                    \
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
DEF_BINARY_OP(^)  // NOLINT
DEF_BINARY_OP(>>)
DEF_BINARY_OP(<<)
}  // End namespace std

using namespace std;  // NOLINT
