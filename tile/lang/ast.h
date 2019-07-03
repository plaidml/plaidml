#pragma once

#include <memory>
#include <string>
#include <vector>

#include "tile/base/shape.h"
#include "tile/lang/compose.h"
#include "tile/lang/type.h"
#include "tile/math/polynomial.h"

namespace vertexai {
namespace tile {
namespace lang {

using Polynomial = math::Polynomial<math::Rational>;

struct Expr;
struct CallExpr;
struct ConstraintExpr;
struct ContractionExpr;
struct DimExprExpr;
struct FloatConst;
struct IntConst;
struct ParamExpr;
struct TensorSpecExpr;
struct TupleExpr;

struct PolyExpr;
struct PolyDimExpr;
struct PolyIndex;
struct PolyLiteral;
struct PolyOpExpr;

struct DimExpr;
struct DimIntExpr;
struct DimNoneExpr;
struct DimOpExpr;
struct DimRefExpr;

struct AstVisitor {
  virtual ~AstVisitor() = default;
  virtual void Visit(const CallExpr& expr) = 0;
  virtual void Visit(const ConstraintExpr& expr) = 0;
  virtual void Visit(const ContractionExpr& expr) = 0;
  virtual void Visit(const DimExprExpr& expr) = 0;
  virtual void Visit(const FloatConst& expr) = 0;
  virtual void Visit(const IntConst& expr) = 0;
  virtual void Visit(const ParamExpr& expr) = 0;
  virtual void Visit(const TensorSpecExpr& expr) = 0;
};

struct LogicalDim {
  std::shared_ptr<DimExpr> expr;

  std::string str() const;
};

struct LogicalShape {
  DataType dtype;
  std::vector<LogicalDim> dims;

  explicit LogicalShape(DataType dtype = DataType::INVALID) : dtype(dtype) {}
  LogicalShape(DataType dtype, const std::vector<std::shared_ptr<DimExpr>>& exprs) : dtype(dtype) {
    for (const auto& dim : exprs) {
      dims.emplace_back(LogicalDim{dim});
    }
  }
  std::string str() const;
  void bind_dims(std::vector<std::shared_ptr<DimExpr>>* into);
};

TensorShape IntoTensorShape(const LogicalShape& shape);

struct Expr {
  std::string name;
  LogicalShape shape;

  explicit Expr(const std::string& name = "") : name(name) {}
  explicit Expr(const LogicalShape& shape, const std::string& name = "") : name(name), shape(shape) {}
  virtual ~Expr() = default;
  virtual void Accept(AstVisitor*) = 0;
  virtual std::string str() const = 0;
};

struct ParamExpr : Expr {
  explicit ParamExpr(const std::string& name = "");
  void Accept(AstVisitor* visitor) { visitor->Visit(*this); }
  std::string str() const;
  void ComputeShape(const std::shared_ptr<ParamExpr>& ref, const LogicalShape& shape);
};

struct IntConst : Expr {
  int64_t value;

  explicit IntConst(int64_t value);
  void Accept(AstVisitor* visitor) { visitor->Visit(*this); }
  std::string str() const;
};

struct FloatConst : Expr {
  double value;

  explicit FloatConst(double value);
  void Accept(AstVisitor* visitor) { visitor->Visit(*this); }
  std::string str() const;
};

struct NoneExpr : Expr {
  NoneExpr() = default;
  void Accept(AstVisitor* visitor) {}
  std::string str() const { return "None"; }
};

struct TupleExpr : Expr {
  std::vector<std::shared_ptr<Expr>> exprs;

  explicit TupleExpr(const std::vector<std::shared_ptr<Expr>>& exprs);
  void Accept(AstVisitor* visitor) {}
  std::string str() const;
};

std::shared_ptr<Expr> MakeCall(const std::string& fn, const std::vector<std::shared_ptr<Expr>>& args);

struct CallExpr : Expr {
  std::string fn;
  std::vector<std::shared_ptr<Expr>> args;

  CallExpr(const std::string& fn, const std::vector<std::shared_ptr<Expr>>& args);
  void Accept(AstVisitor* visitor) { visitor->Visit(*this); }
  std::string str() const;
  void ComputeShape();
};

struct DimExprExpr : Expr {
  std::shared_ptr<DimExpr> expr;

  explicit DimExprExpr(const std::shared_ptr<DimExpr>& expr);
  void Accept(AstVisitor* visitor) { visitor->Visit(*this); }
  std::string str() const;
};

struct TensorSpecExpr : Expr {
  std::shared_ptr<Expr> ref;
  std::vector<std::shared_ptr<PolyExpr>> index_spec;
  std::vector<std::shared_ptr<DimExpr>> output_sizes;

  TensorSpecExpr(const std::shared_ptr<Expr>& ref,                          //
                 const std::vector<std::shared_ptr<PolyExpr>>& index_spec,  //
                 const std::vector<std::shared_ptr<DimExpr>>& output_sizes);
  void Accept(AstVisitor* visitor) { visitor->Visit(*this); }
  std::string str() const;
};

struct ConstraintExpr : Expr {
  std::shared_ptr<PolyExpr> lhs;
  std::shared_ptr<DimExpr> rhs;

  ConstraintExpr(const std::shared_ptr<PolyExpr>& lhs, const std::shared_ptr<DimExpr>& rhs);
  void Accept(AstVisitor* visitor) { visitor->Visit(*this); }
  std::string str() const;
};

struct ContractionExpr : Expr {
  AggregationOp agg_op;
  CombinationOp combo_op;
  std::shared_ptr<TensorSpecExpr> output;
  std::vector<std::shared_ptr<TensorSpecExpr>> inputs;
  std::vector<std::shared_ptr<ConstraintExpr>> constraints;
  bool no_defract = false;
  std::shared_ptr<Expr> use_default;

  ContractionExpr();
  void Accept(AstVisitor* visitor) { visitor->Visit(*this); }
  std::string str() const;
  void ComputeShape();
};

struct PolyVisitor {
  virtual ~PolyVisitor() = default;
  virtual Polynomial Visit(const PolyDimExpr&) = 0;
  virtual Polynomial Visit(const PolyIndex&) = 0;
  virtual Polynomial Visit(const PolyLiteral&) = 0;
  virtual Polynomial Visit(const PolyOpExpr&) = 0;
};

struct PolyExpr {
  virtual ~PolyExpr() = default;
  virtual Polynomial Accept(PolyVisitor*) = 0;
  virtual std::string str() const = 0;
};

struct PolyDimExpr : PolyExpr {
  std::shared_ptr<DimExpr> expr;

  explicit PolyDimExpr(const std::shared_ptr<DimExpr>& expr) : expr(expr) {}
  Polynomial Accept(PolyVisitor* visitor) { return visitor->Visit(*this); }
  std::string str() const;
};

struct PolyIndex : PolyExpr {
  size_t idx_id;
  std::string name;
  mutable std::vector<std::shared_ptr<ConstraintExpr>> constraints;

  explicit PolyIndex(size_t idx_id, const std::string& name = "") : idx_id(idx_id), name(name) {}
  Polynomial Accept(PolyVisitor* visitor) { return visitor->Visit(*this); }
  std::string str() const;
};

struct PolyLiteral : PolyExpr {
  int64_t value;

  explicit PolyLiteral(int64_t value) : value(value) {}
  Polynomial Accept(PolyVisitor* visitor) { return visitor->Visit(*this); }
  std::string str() const;
};

enum class IntOp {
  Neg,
  Add,
  Sub,
  Mul,
  Div,
};

std::shared_ptr<PolyExpr> MakeOp(IntOp op, const std::vector<std::shared_ptr<PolyExpr>>& operands);

struct PolyOpExpr : PolyExpr {
  IntOp op;
  std::vector<std::shared_ptr<PolyExpr>> operands;

  PolyOpExpr(IntOp op, const std::vector<std::shared_ptr<PolyExpr>>& operands) : op(op), operands(operands) {}
  Polynomial Accept(PolyVisitor* visitor) { return visitor->Visit(*this); }
  std::string str() const;
};

struct DimVisitor {
  virtual ~DimVisitor() = default;
  virtual int64_t Visit(const DimIntExpr&) = 0;
  virtual int64_t Visit(const DimOpExpr&) = 0;
  virtual int64_t Visit(const DimNoneExpr&) = 0;
  virtual int64_t Visit(const DimRefExpr&) = 0;
};

struct DimExpr {
  virtual ~DimExpr() = default;
  virtual int64_t Accept(DimVisitor*) = 0;
  virtual std::string str() const = 0;
};

struct DimIntExpr : DimExpr {
  int64_t value;

  explicit DimIntExpr(int64_t value) : value(value) {}
  int64_t Accept(DimVisitor* visitor) { return visitor->Visit(*this); }
  std::string str() const;
};

std::shared_ptr<DimExpr> MakeOp(IntOp op, const std::vector<std::shared_ptr<DimExpr>>& operands);

struct DimOpExpr : DimExpr {
  IntOp op;
  std::vector<std::shared_ptr<DimExpr>> operands;

  DimOpExpr(IntOp op, const std::vector<std::shared_ptr<DimExpr>>& operands) : op(op), operands(operands) {}
  int64_t Accept(DimVisitor* visitor) { return visitor->Visit(*this); }
  std::string str() const;
};

struct DimNoneExpr : DimExpr {
  int64_t Accept(DimVisitor* visitor) { return visitor->Visit(*this); }
  std::string str() const { return "None"; }
};

struct DimRefExpr : DimExpr {
  std::shared_ptr<Expr> ref;
  size_t dim;

  DimRefExpr(const std::shared_ptr<Expr>& ref, size_t dim) : ref(ref), dim(dim) {}
  int64_t Accept(DimVisitor* visitor) { return visitor->Visit(*this); }
  std::string str() const;
};

// This is necessary to allow for these kinds of expressions:
//   if (i - j < 10) {}
//
// What we want is for both `i` and `j` to refer to the `i - j < 10` constraint.
// Later, the ConstraintCollector will track each constraint that is associated
// with the indexes that are in turn associated with a contraction.
class ConstraintApplier : public PolyVisitor {
 public:
  explicit ConstraintApplier(const std::shared_ptr<ConstraintExpr>& constraint) : constraint_(constraint) {}

 private:
  Polynomial Visit(const PolyDimExpr& expr) { return Polynomial(); }

  Polynomial Visit(const PolyIndex& expr) {
    expr.constraints.emplace_back(constraint_);
    return Polynomial();
  }

  Polynomial Visit(const PolyLiteral& expr) { return Polynomial(); }

  Polynomial Visit(const PolyOpExpr& expr) {
    for (auto operand : expr.operands) {
      operand->Accept(this);
    }
    return Polynomial();
  }

 private:
  std::shared_ptr<ConstraintExpr> constraint_;
};

// Add each unique constraint on indexes associated with a contraction.
// Duplicates may occur in cases like:
//   if (i - j < 10) {}
//
// Both `i` and `j` will refer to the same `i - j < 10` constraint.
struct ConstraintCollector : public PolyVisitor {
  Polynomial Visit(const PolyDimExpr& expr) { return Polynomial(); }

  Polynomial Visit(const PolyIndex& expr) {
    for (const auto& constraint : expr.constraints) {
      auto it = std::find(constraints.begin(), constraints.end(), constraint);
      if (it == constraints.end()) {
        constraints.emplace_back(constraint);
      }
    }
    return Polynomial();
  }

  Polynomial Visit(const PolyLiteral& expr) { return Polynomial(); }

  Polynomial Visit(const PolyOpExpr& expr) {
    for (const auto& op : expr.operands) {
      op->Accept(this);
    }
    return Polynomial();
  }

  std::vector<std::shared_ptr<ConstraintExpr>> constraints;
};

TensorShape EvaluateShape(const std::shared_ptr<Expr>& expr);

struct ProgramEvaluation {
  RunInfo runinfo;
  std::vector<const Expr*> inputs;
  std::vector<const Expr*> outputs;
};

ProgramEvaluation Evaluate(const std::string& name, const std::vector<std::shared_ptr<Expr>>& outputs);

}  // namespace lang
}  // namespace tile
}  // namespace vertexai
