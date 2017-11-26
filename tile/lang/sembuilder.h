
#pragma once

#include <memory>
#include <string>
#include <utility>
#include <vector>

#include "tile/lang/semtree.h"

namespace vertexai {
namespace tile {
namespace sem {
namespace builder {

// A DSL for building semantic trees

class LValueHolder {
 public:
  explicit LValueHolder(LValPtr v) : v_(std::move(v)) {}
  operator ExprPtr() const { return std::make_shared<Expression>(LoadExpr(v_)); }
  StmtPtr operator=(const LValueHolder& rhs) const {
    return std::make_shared<Statement>(StoreStmt(v_, std::make_shared<Expression>(LoadExpr(rhs.v_))));
  }
  StmtPtr operator=(int x) const {
    return std::make_shared<Statement>(StoreStmt(v_, std::make_shared<Expression>(IntConst(x))));
  }
  StmtPtr operator=(ExprPtr rhs) const { return std::make_shared<Statement>(StoreStmt(v_, rhs)); }
  LValueHolder operator[](ExprPtr offset) const {
    return LValueHolder(std::make_shared<LValue>(SubscriptLVal(v_, offset)));
  }
  ExprPtr operator()(ExprPtr val) const {
    return std::make_shared<Expression>(
        CallExpr(std::make_shared<Expression>(LoadExpr(v_)), std::vector<ExprPtr>{val}));
  }
  ExprPtr operator()(ExprPtr v1, ExprPtr v2) const {
    return std::make_shared<Expression>(
        CallExpr(std::make_shared<Expression>(LoadExpr(v_)), std::vector<ExprPtr>{v1, v2}));
  }
  ExprPtr operator()(ExprPtr v1, ExprPtr v2, ExprPtr v3) const {
    return std::make_shared<Expression>(
        CallExpr(std::make_shared<Expression>(LoadExpr(v_)), std::vector<ExprPtr>{v1, v2, v3}));
  }
  ExprPtr operator()(int val) const {
    return std::make_shared<Expression>(CallExpr(std::make_shared<Expression>(LoadExpr(v_)),
                                                 std::vector<ExprPtr>({std::make_shared<Expression>(IntConst(val))})));
  }

 private:
  LValPtr v_;
};

inline LValueHolder _(const std::string& name) { return LValueHolder(std::make_shared<LValue>(LookupLVal(name))); }

inline ExprPtr _LimitConst(const LimitConst::Which& which, const lang::DataType& type) {
  return std::make_shared<Expression>(LimitConst(which, type));
}

inline ExprPtr _Index(const IndexExpr::Type& type, size_t dim) {
  return std::make_shared<Expression>(IndexExpr(type, dim));
}

inline BlockPtr _Block(std::initializer_list<StmtPtr> inner) {
  return std::make_shared<Block>(std::vector<StmtPtr>(inner));
}

inline StmtPtr _Stmt(BlockPtr b) { return std::make_shared<Statement>(*b); }

inline StmtPtr _If(ExprPtr cond, BlockPtr iftrue) {
  return std::make_shared<Statement>(IfStmt(cond, iftrue, BlockPtr()));
}
inline StmtPtr _If(ExprPtr cond, StmtPtr iftrue) {
  return std::make_shared<Statement>(IfStmt(cond, _Block({iftrue}), BlockPtr()));
}
inline StmtPtr _If(ExprPtr cond, BlockPtr iftrue, BlockPtr iffalse) {
  return std::make_shared<Statement>(IfStmt(cond, iftrue, iffalse));
}
inline StmtPtr _If(ExprPtr cond, StmtPtr iftrue, StmtPtr iffalse) {
  return std::make_shared<Statement>(IfStmt(cond, _Block({iftrue}), _Block({iffalse})));
}

inline StmtPtr _While(ExprPtr cond, BlockPtr inner) { return std::make_shared<Statement>(WhileStmt(cond, inner)); }

inline StmtPtr _For(const std::string& var, uint64_t n, uint64_t s, BlockPtr inner) {
  return std::make_shared<Statement>(ForStmt(var, n, s, inner));
}

inline StmtPtr _Barrier() { return std::make_shared<Statement>(BarrierStmt()); }
inline StmtPtr _Return(ExprPtr value = ExprPtr()) { return std::make_shared<Statement>(ReturnStmt(value)); }

inline std::shared_ptr<Function> _Function(std::string name, const Type& ret,
                                           std::initializer_list<Function::param_t> params,
                                           std::initializer_list<StmtPtr> body) {
  return std::make_shared<Function>(std::move(name), ret, std::vector<Function::param_t>(params),
                                    std::make_shared<Block>(std::vector<StmtPtr>(body)));
}

template <typename T>
inline std::shared_ptr<Expression> _Const(T x) {
  return std::make_shared<Expression>(IntConst(x));
}
inline std::shared_ptr<Expression> _Const(double x) { return std::make_shared<Expression>(FloatConst(x)); }
inline std::shared_ptr<Statement> _Declare(const Type& type, const std::string& name, ExprPtr init) {
  return std::make_shared<Statement>(DeclareStmt(type, name, init));
}
template <typename T>
inline std::shared_ptr<Statement> _DeclareConst(const Type& type, const std::string& name, T init) {
  return _Declare(type, name, _Const(init));
}
inline std::shared_ptr<Expression> _Cast(const Type& type, ExprPtr init) {
  return std::make_shared<Expression>(CastExpr(type, init));
}

inline std::shared_ptr<Expression> _Cond(ExprPtr cond, ExprPtr tcase, ExprPtr fcase) {
  return std::make_shared<Expression>(CondExpr(cond, tcase, fcase));
}
inline std::shared_ptr<Expression> _Select(ExprPtr cond, ExprPtr tcase, ExprPtr fcase) {
  return std::make_shared<Expression>(SelectExpr(cond, tcase, fcase));
}
inline std::shared_ptr<Expression> _Clamp(ExprPtr val, ExprPtr min, ExprPtr max) {
  return std::make_shared<Expression>(ClampExpr(val, min, max));
}

}  // namespace builder
}  // namespace sem
}  // namespace tile
}  // namespace vertexai

/* Import the ops into the std namespace to use ADL on std::shared_ptr */

// Define some unary ops
#define DEF_UNARY_OP(op)                                                                                  \
  inline vertexai::tile::sem::ExprPtr operator op(vertexai::tile::sem::ExprPtr inner) {                   \
    return std::make_shared<vertexai::tile::sem::Expression>(vertexai::tile::sem::UnaryExpr(#op, inner)); \
  }

// Define some binary ops
#define DEF_BINARY_OP(op)                                                                                           \
  inline vertexai::tile::sem::ExprPtr operator op(vertexai::tile::sem::ExprPtr lhs,                                 \
                                                  vertexai::tile::sem::ExprPtr rhs) {                               \
    return std::make_shared<vertexai::tile::sem::Expression>(vertexai::tile::sem::BinaryExpr(#op, lhs, rhs));       \
  }                                                                                                                 \
  template <class T, typename std::enable_if<std::is_integral<T>::value>::type* = nullptr>                          \
  inline vertexai::tile::sem::ExprPtr operator op(vertexai::tile::sem::ExprPtr lhs, T rhs) {                        \
    auto rhs_const = vertexai::tile::sem::builder::_Const(rhs);                                                     \
    return std::make_shared<vertexai::tile::sem::Expression>(vertexai::tile::sem::BinaryExpr(#op, lhs, rhs_const)); \
  }                                                                                                                 \
  template <class T, typename std::enable_if<std::is_integral<T>::value>::type* = nullptr>                          \
  inline vertexai::tile::sem::ExprPtr operator op(T lhs, vertexai::tile::sem::ExprPtr rhs) {                        \
    auto lhs_const = vertexai::tile::sem::builder::_Const(lhs);                                                     \
    return std::make_shared<vertexai::tile::sem::Expression>(vertexai::tile::sem::BinaryExpr(#op, lhs_const, rhs)); \
  }                                                                                                                 \
  inline vertexai::tile::sem::ExprPtr operator op(vertexai::tile::sem::ExprPtr lhs, double rhs) {                   \
    auto rhs_const = vertexai::tile::sem::builder::_Const(rhs);                                                     \
    return std::make_shared<vertexai::tile::sem::Expression>(vertexai::tile::sem::BinaryExpr(#op, lhs, rhs_const)); \
  }                                                                                                                 \
  inline vertexai::tile::sem::ExprPtr operator op(double lhs, vertexai::tile::sem::ExprPtr rhs) {                   \
    auto lhs_const = vertexai::tile::sem::builder::_Const(lhs);                                                     \
    return std::make_shared<vertexai::tile::sem::Expression>(vertexai::tile::sem::BinaryExpr(#op, lhs_const, rhs)); \
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
