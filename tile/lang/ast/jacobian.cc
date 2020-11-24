// Copyright 2019 Intel Corporation.

#include "tile/lang/ast/jacobian.h"

#include <map>
#include <memory>
#include <stack>
#include <unordered_set>

#include <boost/format.hpp>

#include "tile/lang/ast/gradient.h"
#include "tile/lang/ast/traversal.h"

namespace vertexai {
namespace tile {
namespace lang {
namespace ast {

namespace {

struct UseInfo {
  ExprPtr expr;
  size_t idx;
};

class ComputeUses : public AstVisitor<void> {
 public:
  explicit ComputeUses(const ExprPtr& src) {
    stack_.push(src);
    while (stack_.size()) {
      auto expr = stack_.top();
      stack_.pop();
      if (!seen_.count(expr.get())) {
        seen_.insert(expr.get());
        expr->Accept(this);
      }
    }
  }

  const std::vector<UseInfo>& uses(const Expr* expr) const { return uses_.at(expr); }

 private:
  void Visit(const CallExpr& expr) final {
    for (size_t i = 0; i < expr.args.size(); i++) {
      Push(expr, expr.args[i], i);
    }
  }

  void Visit(const ContractionExpr& expr) final {
    for (size_t i = 0; i < expr.srcs.size(); i++) {
      Push(expr, expr.srcs[i]->ref, i);
    }
    if (expr.use_default) {
      Push(expr, expr.use_default, expr.srcs.size());
    }
  }

  void Visit(const GradOverrideExpr& expr) final {
    for (size_t i = 0; i < expr.ins.size(); i++) {
      Push(expr, expr.ins[i], i);
    }
  }

  void Visit(const DimExprExpr& expr) final {}
  void Visit(const FloatConst& expr) final {}
  void Visit(const IntConst& expr) final {}
  void Visit(const ParamExpr& expr) final {}

 private:
  void Push(const Expr& user, const ExprPtr& used, size_t idx) {
    IVLOG(2, "ComputeUses::Push> user: " << &user << ", used: " << used << ", idx: " << idx);
    auto ptr = std::const_pointer_cast<Expr>(user.as_ptr());
    uses_[used.get()].emplace_back(UseInfo{ptr, idx});
    stack_.push(used);
  }

 private:
  std::stack<ExprPtr> stack_;
  std::unordered_set<const Expr*> seen_;
  std::unordered_map<const Expr*, std::vector<UseInfo>> uses_;
};

class Jacobian {
 public:
  explicit Jacobian(const ExprPtr& err) : uses_(err) {
    IVLOG(2, "Jacobian::Jacobian> err: " << err);

    // Create identity matrix to represent d(err)/d(err)
    auto dims = err->shape.dims_as_exprs();
    orank_ = dims.size();

    auto Ji = std::make_shared<ContractionExpr>();
    Ji->agg_op = AggregationOp::ASSIGN;
    Ji->combo_op = CombinationOp::NONE;

    auto src = std::make_shared<FloatConst>(1.0);
    Ji->srcs.push_back(std::make_shared<IndexMapExpr>(src, std::vector<PolyExprPtr>{}));

    std::vector<PolyExprPtr> Jidxs;
    std::vector<DimExprPtr> Jdims;

    for (size_t io = 0; io < 2; io++) {
      for (size_t i = 0; i < orank_; i++) {
        Jidxs.push_back(std::make_shared<PolyIndex>(i));
        Jdims.push_back(dims[i]);
      }
    }

    Ji->sink_idxs = std::make_shared<IndexMapExpr>(nullptr, Jidxs);
    Ji->sink_dims = std::make_shared<SizeMapExpr>(Jdims);
    Ji->ComputeShape("");

    seen_[err.get()] = Ji;
  }

  ExprPtr GetDerivative(const ExprPtr& expr) {
    IVLOG(2, "Jacobian::GetDerivative> " << expr);
    auto it = seen_.find(expr.get());
    if (it != seen_.end()) {
      IVLOG(2, "  returning: " << it->second);
      return it->second;
    }
    ExprPtr total;
    for (const auto& use : uses_.uses(expr.get())) {
      ExprPtr dop;
      auto dout = GetDerivative(use.expr);
      if (auto grad_override_expr = std::dynamic_pointer_cast<GradOverrideExpr>(use.expr)) {
        // A gradient override replaces all the derivatives, so set total and exit the loop
        total = DeriveOverride(dout, grad_override_expr, use.idx);
        break;
      }
      if (auto call_expr = std::dynamic_pointer_cast<CallExpr>(use.expr)) {
        dop = DeriveCall(dout, call_expr, use.idx);
      } else if (auto cion_expr = std::dynamic_pointer_cast<ContractionExpr>(use.expr)) {
        dop = DeriveContraction(dout, cion_expr, use.idx);
      } else {
        throw std::runtime_error("Invalid operation type in Gradient::GetDerivative");
      }
      if (!total) {
        total = dop;
      } else {
        total = MakeCall("add", {total, dop});
      }
    }
    if (!total) {
      total = std::make_shared<FloatConst>(0.0);
    }
    IVLOG(2, "  Gradient::GetDerivative, final result -> " << total);
    seen_.emplace(expr.get(), total);
    return total;
  }

 private:
  ExprPtr DeriveContraction(const ExprPtr& dout, const std::shared_ptr<ContractionExpr>& expr, size_t idx) {
    if (expr->use_default && idx == expr->srcs.size()) {
      return dout;
    }
    if (expr->combo_op == CombinationOp::EQ) {
      return std::make_shared<IntConst>(0);
    }
    if (expr->agg_op == AggregationOp::SUM || expr->agg_op == AggregationOp::ASSIGN) {
      return DeriveSum(dout, expr, idx);
    }
    if (expr->agg_op == AggregationOp::MIN || expr->agg_op == AggregationOp::MAX) {
      return DeriveExtreme(dout, expr, idx);
    }
    if (expr->agg_op == AggregationOp::PROD) {
      throw std::runtime_error("PROD AggregationOp not implemented for Jacobian");
    }
    throw std::runtime_error("Invalid ContractionExpr in DeriveContraction");
  }

  // Each of these must compute single-layer Jacobian of op, then combine it with prior Jacobian dout to
  // produce new total Jacobian

  ExprPtr DeriveCall(const ExprPtr& dout, const std::shared_ptr<CallExpr>& op, size_t idx) {
    if (op->fn == "reshape") {
      throw std::runtime_error("Jacobian not implemented for reshape");
    }
    auto deriv = DerivRegistry::Instance()->Resolve(op->fn);  // Returns elementwise derivative data
    // Autobroadcasting handles J i/o dims correctly
    return deriv.fn(op, dout, op->args, deriv.user_fn, deriv.user_ctx)[idx];
  }

  ExprPtr DeriveSum(const ExprPtr& dout, const std::shared_ptr<ContractionExpr>& op, size_t idx) {
    // Compute Jacobian for SumContraction Op
    auto Ji = std::make_shared<ContractionExpr>();
    Ji->agg_op = AggregationOp::ASSIGN;
    Ji->constraints = op->constraints;

    std::vector<PolyExprPtr> Jidxs;  // Indexes of new total Jacobian
    std::vector<DimExprPtr> Jdims;   // Dimensions of new total Jacobian

    // Add dimensions to Ji corresponding to op output dimensions
    auto odims = op->shape.dims_as_exprs();
    for (size_t i = 0; i < odims.size(); i++) {
      Jidxs.push_back(op->sink_idxs->idxs[i]);
      Jdims.push_back(odims[i]);
    }

    for (size_t j = 0; j < op->srcs.size(); j++) {
      if (j == idx) {
        // Add dimensions to Ji corresponding to op w.r.t. input dimensions
        auto idims = op->srcs[j]->ref->shape.dims_as_exprs();
        for (size_t i = 0; i < idims.size(); i++) {
          Jidxs.push_back(op->srcs[j]->idxs[i]);
          Jdims.push_back(idims[i]);
        }
      } else {
        switch (op->combo_op) {
          case CombinationOp::MULTIPLY:
            // Add non-w.r.t. op inputs as inputs to Ji, set combo op for Ji
            Ji->srcs.push_back(op->srcs[j]);
            Ji->combo_op = CombinationOp::MULTIPLY;
            break;
          case CombinationOp::PLUS:
            // Jacobian is identity matrix, broadcast through non-wrt dimensions
            Ji->srcs.push_back(
                std::make_shared<IndexMapExpr>(std::make_shared<FloatConst>(1.0), std::vector<PolyExprPtr>{}));
            Ji->combo_op = CombinationOp::NONE;
            break;
          default:
            throw std::runtime_error("Combination Op receieved by DeriveSum in Jacobian not implemented");
        }
      }
    }

    Ji->sink_idxs = std::make_shared<IndexMapExpr>(nullptr, Jidxs);
    Ji->sink_dims = std::make_shared<SizeMapExpr>(Jdims);
    Ji->ComputeShape("");

    return ChainRule(dout, Ji);
  }

  ExprPtr DeriveOverride(const ExprPtr& dout, const std::shared_ptr<GradOverrideExpr>& op, size_t idx) {
    throw std::runtime_error("DeriveOverride not implemented for Jacobian");
  }

  ExprPtr DeriveExtreme(const ExprPtr& dout, const std::shared_ptr<ContractionExpr>& op, size_t idx) {
    throw std::runtime_error("DeriveExtreme not implemented for Jacobian");
  }

  ExprPtr ChainRule(const ExprPtr& Jprev, const ExprPtr& Ji) {
    // Combine current Jacobian with total. Output dimensions are
    // [first {orank_} dimensions of Jprev, last {Ji_rank - orank} dimensions of Ji]
    auto Jnew = std::make_shared<ContractionExpr>();
    Jnew->agg_op = AggregationOp::SUM;
    Jnew->combo_op = CombinationOp::MULTIPLY;

    std::vector<PolyExprPtr> iidxs;  // Indices for op Jacobian
    std::vector<PolyExprPtr> pidxs;  // Indices for previous total Jacobian
    std::vector<PolyExprPtr> nidxs;  // Indices for new total Jacobian

    auto idims = Ji->shape.dims_as_exprs();     // Dimensions for op Jacobian
    auto pdims = Jprev->shape.dims_as_exprs();  // Dimensions for previous total Jacobian
    std::vector<DimExprPtr> ndims;              // Dimensions for new total Jacobian

    size_t Jirank = idims.size();
    size_t Jprank = pdims.size();

    // Add dims/idxs corresponding to output dimensions
    for (size_t i = 0; i < orank_; i++) {
      auto idx = std::make_shared<PolyIndex>(i);
      pidxs.push_back(idx);
      nidxs.push_back(idx);
      ndims.push_back(pdims[i]);
    }

    // Add "overlapping" idxs
    for (size_t i = orank_; i < Jprank; i++) {
      auto idx = std::make_shared<PolyIndex>(i);
      pidxs.push_back(idx);
      iidxs.push_back(idx);
    }

    // Add dim/idxs corresponding to new input dimensions
    for (size_t i = Jprank - orank_; i < Jirank; i++) {
      auto idx = std::make_shared<PolyIndex>(i + orank_);
      iidxs.push_back(idx);
      nidxs.push_back(idx);
      ndims.push_back(idims[i]);
    }

    // Bind sources
    Jnew->srcs.push_back(std::make_shared<IndexMapExpr>(Jprev, pidxs));
    Jnew->srcs.push_back(std::make_shared<IndexMapExpr>(Ji, iidxs));

    // Bind output dims/idxs
    Jnew->sink_idxs = std::make_shared<IndexMapExpr>(nullptr, nidxs);
    Jnew->sink_dims = std::make_shared<SizeMapExpr>(ndims);
    Jnew->ComputeShape("");

    return Jnew;
  }

 private:
  ComputeUses uses_;
  std::map<const Expr*, ExprPtr> seen_;
  size_t orank_;
};

}  // namespace

std::vector<ExprPtr> ComputeJacobian(const std::vector<ExprPtr>& wrts, const ExprPtr& loss) {
  ExprPtr value = loss;
  Jacobian grad(value);
  std::vector<ExprPtr> ret(wrts.size());
  for (size_t i = 0; i < wrts.size(); i++) {
    auto wrt = wrts[i];
    ret[i] = grad.GetDerivative(wrts[i]);
  }
  return ret;
}

}  // namespace ast
}  // namespace lang
}  // namespace tile
}  // namespace vertexai
