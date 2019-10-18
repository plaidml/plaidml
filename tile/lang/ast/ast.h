// Copyright 2019 Intel Corporation.

#pragma once

#include <memory>
#include <string>
#include <unordered_map>
#include <vector>

#include "tile/base/buffer.h"
#include "tile/lang/runinfo.h"
#include "tile/lang/type.h"
#include "tile/math/polynomial.h"

namespace vertexai {
namespace tile {
namespace lang {
namespace ast {

using Polynomial = math::Polynomial<math::Rational>;

struct Expr;
struct CallExpr;
struct ConstraintExpr;
struct ContractionExpr;
struct DimExprExpr;
struct FloatConst;
struct IntConst;
struct ParamExpr;
struct StringExpr;
struct IndexMapExpr;
struct SizeMapExpr;
struct TupleExpr;
struct GradOverrideExpr;

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

using ExprPtr = std::shared_ptr<Expr>;
using DimExprPtr = std::shared_ptr<DimExpr>;
using PolyExprPtr = std::shared_ptr<PolyExpr>;

// ExprDeriv is a function that builds the gradient ("dX") from the following inputs:
//  1. Y: The Expr for the node
//  2. dY: The Expr for the already-computed gradient of the node's output
//  3. Xs: The Exprs for the node's inputs
using ExprDeriv = std::function<std::vector<ExprPtr>(  //
    const ExprPtr& Y,                                  //
    const ExprPtr& dY,                                 //
    const std::vector<ExprPtr>& Xs,                    //
    void* user_fn,                                     //
    void* user_ctx)>;

// An ExprDerivEntry bundles the ExprDeriv with the FFI-processed user function & context needed to call it from Exprs
struct ExprDerivEntry {
  ExprDeriv fn;
  void* user_fn;
  void* user_ctx;
};

template <typename T>
struct AstVisitor {
  virtual ~AstVisitor() = default;
  virtual T Visit(const CallExpr& expr) = 0;
  virtual T Visit(const ContractionExpr& expr) = 0;
  virtual T Visit(const DimExprExpr& expr) = 0;
  virtual T Visit(const FloatConst& expr) = 0;
  virtual T Visit(const IntConst& expr) = 0;
  virtual T Visit(const ParamExpr& expr) = 0;
  virtual T Visit(const GradOverrideExpr& expr) = 0;
};

struct LogicalDim {
  DimExprPtr expr;

  std::string str() const;
};

struct LogicalShape {
  DataType dtype;
  std::string layout;
  std::vector<LogicalDim> dims;

  explicit LogicalShape(DataType dtype = DataType::INVALID, const std::string& layout = "")
      : dtype(dtype), layout(layout) {}

  LogicalShape(DataType dtype, const std::vector<DimExprPtr>& exprs) : dtype(dtype) {
    for (const auto& dim : exprs) {
      dims.emplace_back(LogicalDim{dim});
    }
  }

  std::string str() const;
  void bind_dims(std::vector<DimExprPtr>* into);
  std::vector<DimExprPtr> dims_as_exprs() const;
};

TensorShape IntoTensorShape(const LogicalShape& shape);

struct Expr : std::enable_shared_from_this<Expr> {
  std::string name;
  LogicalShape shape;

  explicit Expr(const std::string& name = "") : name(name) {}
  explicit Expr(const LogicalShape& shape, const std::string& name = "") : name(name), shape(shape) {}
  virtual ~Expr() = default;
  virtual void Accept(AstVisitor<void>*) = 0;
  virtual std::string str() const = 0;

  std::shared_ptr<Expr> as_ptr() { return shared_from_this(); }
  std::shared_ptr<const Expr> as_ptr() const { return shared_from_this(); }
};

struct ParamExpr : Expr {
  std::shared_ptr<tile::Buffer> buffer;
  explicit ParamExpr(const std::string& name = "");
  void Accept(AstVisitor<void>* visitor) { visitor->Visit(*this); }
  std::string str() const;
  void ComputeShape(const std::shared_ptr<ParamExpr>& ref, const LogicalShape& shape);
};

struct IntConst : Expr {
  int64_t value;

  explicit IntConst(int64_t value);
  void Accept(AstVisitor<void>* visitor) { visitor->Visit(*this); }
  std::string str() const;
};

struct FloatConst : Expr {
  double value;

  explicit FloatConst(double value);
  void Accept(AstVisitor<void>* visitor) { visitor->Visit(*this); }
  std::string str() const;
};

struct StringExpr : Expr {
  std::string value;

  explicit StringExpr(const std::string& value) : value(value) {}
  void Accept(AstVisitor<void>* visitor) {}
  std::string str() const { return "\"" + value + "\""; }
};

struct NoneExpr : Expr {
  NoneExpr() = default;
  void Accept(AstVisitor<void>* visitor) {}
  std::string str() const { return "None"; }
};

struct TupleExpr : Expr {
  std::vector<ExprPtr> exprs;

  explicit TupleExpr(const std::vector<ExprPtr>& exprs);
  void Accept(AstVisitor<void>* visitor) {}
  std::string str() const;
};

ExprPtr MakeGradOverride(const std::shared_ptr<ExprDerivEntry>& fn, const std::vector<ExprPtr>& ins,
                         const ExprPtr& out);

// TODO: Consider adding this lookup function to grab a gradient from the deriv registry:
// ExprPtr MakeGradOverride(const std::string& fn_name, const std::vector<ExprPtr>& ins, const ExprPtr& out);

struct GradOverrideExpr : Expr {
  std::shared_ptr<ExprDerivEntry> fn;
  std::vector<ExprPtr> ins;  // These will have their derivatives overridden
  ExprPtr out;               // This will passthrough as the forward pass output; used as Y in calls to fn

  GradOverrideExpr(const std::shared_ptr<ExprDerivEntry>& fn, const std::vector<ExprPtr>& ins, const ExprPtr& out);
  void Accept(AstVisitor<void>* visitor) { visitor->Visit(*this); }
  std::string str() const;
  void ComputeShape();
};

ExprPtr MakeCall(const std::string& fn, const std::vector<ExprPtr>& args);

struct CallExpr : Expr {
  std::string fn;
  std::vector<ExprPtr> args;

  CallExpr(const std::string& fn, const std::vector<ExprPtr>& args);
  void Accept(AstVisitor<void>* visitor) { visitor->Visit(*this); }
  std::string str() const;
  void ComputeShape();
};

struct DimExprExpr : Expr {
  DimExprPtr expr;

  explicit DimExprExpr(const DimExprPtr& expr);
  void Accept(AstVisitor<void>* visitor) { visitor->Visit(*this); }
  std::string str() const;
};

struct IndexMapExpr : Expr {
  ExprPtr ref;
  std::vector<PolyExprPtr> idxs;

  IndexMapExpr(            //
      const ExprPtr& ref,  //
      const std::vector<PolyExprPtr>& idxs);

  void Accept(AstVisitor<void>* visitor) {}
  std::string str() const;
};

struct SizeMapExpr : Expr {
  std::vector<DimExprPtr> dims;

  explicit SizeMapExpr(const std::vector<DimExprPtr>& dims);

  void Accept(AstVisitor<void>* visitor) {}
  std::string str() const;
};

struct ConstraintExpr : Expr {
  PolyExprPtr lhs;
  DimExprPtr rhs;

  ConstraintExpr(const PolyExprPtr& lhs, const DimExprPtr& rhs);
  void Accept(AstVisitor<void>* visitor) {}
  std::string str() const;
};

struct ContractionExpr : Expr {
  AggregationOp agg_op;
  CombinationOp combo_op;
  std::shared_ptr<IndexMapExpr> sink_idxs;
  std::shared_ptr<SizeMapExpr> sink_dims;
  std::vector<std::shared_ptr<IndexMapExpr>> srcs;
  std::vector<std::shared_ptr<ConstraintExpr>> constraints;
  bool no_defract = false;
  ExprPtr use_default;

  ContractionExpr();
  void Accept(AstVisitor<void>* visitor) { visitor->Visit(*this); }
  std::string str() const;
  void ComputeShape(const std::string& layout);
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
  DimExprPtr expr;

  explicit PolyDimExpr(const DimExprPtr& expr) : expr(expr) {}
  Polynomial Accept(PolyVisitor* visitor) { return visitor->Visit(*this); }
  std::string str() const;
};

struct PolyIndex : PolyExpr {
  size_t idx_id;
  std::string name;

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
  Max,
  Min,
};

PolyExprPtr MakeOp(IntOp op, const std::vector<PolyExprPtr>& operands);

struct PolyOpExpr : PolyExpr {
  IntOp op;
  std::vector<PolyExprPtr> operands;

  PolyOpExpr(IntOp op, const std::vector<PolyExprPtr>& operands) : op(op), operands(operands) {}
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

DimExprPtr MakeOp(IntOp op, const std::vector<DimExprPtr>& operands);

struct DimOpExpr : DimExpr {
  IntOp op;
  std::vector<DimExprPtr> operands;

  DimOpExpr(IntOp op, const std::vector<DimExprPtr>& operands) : op(op), operands(operands) {}
  int64_t Accept(DimVisitor* visitor) { return visitor->Visit(*this); }
  std::string str() const;
};

struct DimNoneExpr : DimExpr {
  int64_t Accept(DimVisitor* visitor) { return visitor->Visit(*this); }
  std::string str() const { return "None"; }
};

struct DimRefExpr : DimExpr {
  ExprPtr ref;
  size_t dim;

  DimRefExpr(const ExprPtr& ref, size_t dim) : ref(ref), dim(dim) {}
  int64_t Accept(DimVisitor* visitor) { return visitor->Visit(*this); }
  std::string str() const;
};

struct ProgramEvaluation {
  RunInfo runinfo;
  std::vector<const ParamExpr*> inputs;
  std::vector<ExprPtr> outputs;
  std::unordered_map<const Expr*, std::string> names_by_expr;
  std::unordered_map<std::string, const ParamExpr*> updates;
};

struct ProgramUpdate {
  ExprPtr src;
  ExprPtr dst;
};

struct ProgramMutations {
  std::vector<ExprPtr> outputs;
  std::vector<ProgramUpdate> updates;
};

ProgramEvaluation Evaluate(const std::string& name, ProgramMutations mutations);

std::ostream& operator<<(std::ostream& os, const Expr* expr);

}  // namespace ast
}  // namespace lang
}  // namespace tile
}  // namespace vertexai
