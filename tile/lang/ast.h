#pragma once

#include <memory>
#include <string>
#include <vector>

#include <boost/format.hpp>

#include "tile/base/shape.h"
#include "tile/lang/type.h"
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

struct AstVisitor {
  virtual ~AstVisitor() = default;
  virtual void Visit(const CallExpr& expr) = 0;
  virtual void Visit(const ConstraintExpr& expr) = 0;
  virtual void Visit(const ContractionExpr& expr) = 0;
  virtual void Visit(const FloatConst& expr) = 0;
  virtual void Visit(const IntConst& expr) = 0;
  virtual void Visit(const ParamExpr& expr) = 0;
  virtual void Visit(const TensorSpecExpr& expr) = 0;
};

struct Expr {
  std::string name;

  explicit Expr(const std::string& name = "") : name(name) {}
  virtual ~Expr() = default;
  virtual void Accept(AstVisitor*) = 0;
  virtual std::string str() const = 0;
};

struct ParamExpr : Expr {
  TensorShape shape;

  explicit ParamExpr(const TensorShape& shape, const std::string& name = "") : Expr(name), shape(shape) {}
  void Accept(AstVisitor* visitor) { visitor->Visit(*this); }
  std::string str() const { return "ParamExpr"; }
};

struct IntConst : Expr {
  int64_t value;

  explicit IntConst(int64_t value) : value(value) {}
  void Accept(AstVisitor* visitor) { visitor->Visit(*this); }
  std::string str() const { return "IntConst"; }
};

struct FloatConst : Expr {
  double value;

  explicit FloatConst(double value) : value(value) {}
  void Accept(AstVisitor* visitor) { visitor->Visit(*this); }
  std::string str() const { return "FloatConst"; }
};

struct CallExpr : Expr {
  std::string fn;
  std::vector<std::shared_ptr<Expr>> args;

  CallExpr(const std::string& fn, const std::vector<std::shared_ptr<Expr>>& args) : fn(fn), args(args) {}

  void Accept(AstVisitor* visitor) { visitor->Visit(*this); }
  std::string str() const { return boost::str(boost::format("CallExpr(%1%)") % fn); }
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

  void Accept(AstVisitor* visitor) { visitor->Visit(*this); }
  std::string str() const { return "TensorSpecExpr"; }
};

struct ConstraintExpr : Expr {
  std::shared_ptr<PolyExpr> lhs;
  size_t rhs;

  ConstraintExpr(const std::shared_ptr<PolyExpr>& lhs, size_t rhs) : lhs(lhs), rhs(rhs) {}
  void Accept(AstVisitor* visitor) { visitor->Visit(*this); }
  std::string str() const { return "ConstraintExpr"; }
};

struct ContractionExpr : Expr {
  AggregationOp agg_op;
  CombinationOp combo_op;
  std::shared_ptr<TensorSpecExpr> output;
  std::vector<std::shared_ptr<TensorSpecExpr>> inputs;
  std::vector<std::shared_ptr<ConstraintExpr>> constraints;
  bool no_defract = false;
  std::shared_ptr<Expr> use_default;

  void Accept(AstVisitor* visitor) { visitor->Visit(*this); }
  std::string str() const { return "ContractionExpr"; }
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
  std::string name;

  explicit PolyIndex(const void* ptr, const std::string& name = "") : ptr(ptr), name(name) {}
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
