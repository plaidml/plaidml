#pragma once

#include <memory>
#include <string>
#include <vector>

#include "tile/math/polynomial.h"

namespace vertexai {
namespace tile {
namespace lang {

using Polynomial = math::Polynomial<math::Rational>;

struct CallExpr;
struct ConstraintExpr;
struct ContractionExpr;
struct Expr;
struct FloatConst;
struct IntConst;
struct ParamExpr;
struct TensorSpecExpr;

struct PolyExpr;
struct PolyIndex;
struct PolyLiteral;
struct PolyOp;

template <typename T>
struct AstVisitor {
  virtual ~AstVisitor() = default;
  virtual T Visit(const CallExpr&) = 0;
  virtual T Visit(const ConstraintExpr&) = 0;
  virtual T Visit(const ContractionExpr&) = 0;
  virtual T Visit(const FloatConst&) = 0;
  virtual T Visit(const IntConst&) = 0;
  virtual T Visit(const ParamExpr&) = 0;
  virtual T Visit(const TensorSpecExpr&) = 0;
};

struct Expr {
  virtual ~Expr() = default;
  virtual std::string Accept(AstVisitor<std::string>*) = 0;
};

struct ParamExpr : Expr {
  size_t ndims;
  std::string name;

  explicit ParamExpr(size_t ndims, const std::string& name) : ndims(ndims), name(name) {}
  std::string Accept(AstVisitor<std::string>* visitor) { return visitor->Visit(*this); }
};

struct IntConst : Expr {
  int64_t value;

  explicit IntConst(int64_t value) : value(value) {}
  std::string Accept(AstVisitor<std::string>* visitor) { return visitor->Visit(*this); }
};

struct FloatConst : Expr {
  double value;

  explicit FloatConst(double value) : value(value) {}
  std::string Accept(AstVisitor<std::string>* visitor) { return visitor->Visit(*this); }
};

struct CallExpr : Expr {
  std::string fn;
  std::vector<std::shared_ptr<Expr>> args;

  CallExpr(const std::string& fn, const std::vector<std::shared_ptr<Expr>>& args) : fn(fn), args(args) {}

  std::string Accept(AstVisitor<std::string>* visitor) { return visitor->Visit(*this); }
};

struct TensorSpecExpr : Expr {
  std::shared_ptr<Expr> ref;
  std::vector<std::shared_ptr<PolyExpr>> index_spec;
  std::vector<size_t> output_sizes;

  TensorSpecExpr() = default;
  TensorSpecExpr(const std::shared_ptr<Expr>& ref,  //
                 const std::vector<std::shared_ptr<PolyExpr>>& index_spec,
                 const std::vector<size_t>& output_sizes)
      : ref(ref),  //
        index_spec(index_spec),
        output_sizes(output_sizes) {}

  std::string Accept(AstVisitor<std::string>* visitor) { return visitor->Visit(*this); }
};

struct ConstraintExpr : Expr {
  std::shared_ptr<PolyExpr> lhs;
  size_t rhs;

  ConstraintExpr(const std::shared_ptr<PolyExpr>& lhs, size_t rhs) : lhs(lhs), rhs(rhs) {}
  std::string Accept(AstVisitor<std::string>* visitor) { return visitor->Visit(*this); }
};

struct ContractionExpr : Expr {
  AggregationOp agg_op;
  CombinationOp combo_op;
  std::shared_ptr<TensorSpecExpr> output;
  std::vector<std::shared_ptr<TensorSpecExpr>> inputs;
  std::vector<std::shared_ptr<ConstraintExpr>> constraints;
  bool no_defract = false;
  std::shared_ptr<Expr> use_default;

  std::string Accept(AstVisitor<std::string>* visitor) { return visitor->Visit(*this); }
};

struct PolyVisitor {
  virtual ~PolyVisitor() = default;
  virtual Polynomial Visit(const PolyIndex&) = 0;
  virtual Polynomial Visit(const PolyLiteral&) = 0;
  virtual Polynomial Visit(const PolyOp&) = 0;
};

struct PolyExpr {
  virtual ~PolyExpr() = default;
  virtual Polynomial Accept(PolyVisitor*) = 0;
};

struct PolyIndex : PolyExpr {
  const void* ptr;

  explicit PolyIndex(const void* ptr) : ptr(ptr) {}
  Polynomial Accept(PolyVisitor* visitor) { return visitor->Visit(*this); }
};

struct PolyLiteral : PolyExpr {
  int64_t value;

  explicit PolyLiteral(int64_t value) : value(value) {}
  Polynomial Accept(PolyVisitor* visitor) { return visitor->Visit(*this); }
};

struct PolyOp : PolyExpr {
  std::string op;
  std::vector<std::shared_ptr<PolyExpr>> operands;

  PolyOp(const std::string& op, const std::vector<std::shared_ptr<PolyExpr>>& operands) : op(op), operands(operands) {}
  Polynomial Accept(PolyVisitor* visitor) { return visitor->Visit(*this); }
};

}  // namespace lang
}  // namespace tile
}  // namespace vertexai
