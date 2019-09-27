// Copyright 2019 Intel Corporation.

#include "tile/lang/ast/gradient.h"

#include <map>
#include <memory>
#include <stack>
#include <unordered_set>

#include <boost/format.hpp>

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
    throw std::runtime_error("Well, at least we got here");
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
    IVLOG(4, "ComputeUses::Push> user: " << &user << ", used: " << used << ", idx: " << idx);
    auto ptr = std::const_pointer_cast<Expr>(user.as_ptr());
    uses_[used.get()].emplace_back(UseInfo{ptr, idx});
    stack_.push(used);
  }

 private:
  std::stack<ExprPtr> stack_;
  std::unordered_set<const Expr*> seen_;
  std::unordered_map<const Expr*, std::vector<UseInfo>> uses_;
};

class Gradient {
 public:
  explicit Gradient(const ExprPtr& err) : uses_(err) {
    IVLOG(4, "Gradient::Gradient> err: " << err);
    seen_[err.get()] = std::make_shared<FloatConst>(1.0);
  }

  ExprPtr GetDerivative(const ExprPtr& expr) {
    IVLOG(4, "Gradient::GetDerivative> " << expr);
    auto it = seen_.find(expr.get());
    if (it != seen_.end()) {
      IVLOG(5, "  returning: " << it->second);
      return it->second;
    }
    ExprPtr total;
    for (const auto& use : uses_.uses(expr.get())) {
      ExprPtr dop;
      auto dout = GetDerivative(use.expr);
      if (auto grad_override_expr = std::dynamic_pointer_cast<GradOverrideExpr>(use.expr)) {
        // A gradient override replaces all the derivatives, so set total and exit the loop
        throw std::runtime_error("Yes, we did get here!");
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
    } else if (total->shape.dims.size()) {
      total = MakeCall("simple_reduce", {total, expr});
    }
    IVLOG(4, "  Gradient::GetDerivative, final result -> " << total);
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
      throw std::runtime_error("PROD AggregationOp does not support differentiation");
    }
    throw std::runtime_error("Invalid ContractionExpr in DeriveContraction");
  }

  ExprPtr DeriveCall(const ExprPtr& dout, const std::shared_ptr<CallExpr>& op, size_t idx) {
    IVLOG(4, "Gradient::DeriveCall> dout=" << dout << ", op=" << op << ", fn=" << op->fn << ", idx=" << idx);
    if (op->fn == "reshape") {
      std::vector<ExprPtr> args = {dout};
      auto in = op->args[0];
      auto dim_exprs = in->shape.dims_as_exprs();
      for (size_t i = 0; i < in->shape.dims.size(); ++i) {
        args.push_back(std::make_shared<DimExprExpr>(dim_exprs[i]));
      }
      return MakeCall("reshape", args);
    }
    auto deriv = DerivRegistry::Instance()->Resolve(op->fn);
    return deriv.fn(op, dout, op->args, deriv.user_fn, deriv.user_ctx)[idx];
  }

  ExprPtr DeriveSum(const ExprPtr& dout, const std::shared_ptr<ContractionExpr>& op, size_t idx) {
    IVLOG(4, "Gradient::DeriveSum> dout=" << dout << ", op=" << op << ", idx=" << idx);
    auto dop = std::make_shared<ContractionExpr>();
    dop->agg_op = AggregationOp::SUM;
    dop->combo_op = CombinationOp::NONE;  // May be overridden below based on op->combo_op
    dop->constraints = op->constraints;
    // Anywhere the forward pass hits the default, the derivative w.r.t. any other tensor is 0;
    // thus, for the corresponding gradient, the default is everywhere zero i.e. the standard unspecified default
    if (idx == op->srcs.size()) {
      throw std::logic_error("A default tensor fell through to the DeriveSum case during Gradient");
    }
    for (size_t i = 0; i < op->srcs.size(); ++i) {
      if (idx == i) {
        dop->srcs.push_back(std::make_shared<IndexMapExpr>(dout, op->sink_idxs->idxs));
      } else {
        switch (op->combo_op) {
          case CombinationOp::MULTIPLY:
            // For *, we multiply by the other (non-differentiated) input
            dop->srcs.push_back(op->srcs[i]);
            dop->combo_op = CombinationOp::MULTIPLY;
            break;
          case CombinationOp::PLUS:
            // For +, we ignore the other (non-differentiated) input
            dop->combo_op = CombinationOp::NONE;
            break;
          case CombinationOp::COND:
            throw std::runtime_error("Gradient of sum of conditionals not supported");
          case CombinationOp::NONE:
            throw std::runtime_error(
                "Unexpected multiple inputs found when differentiating contraction with NONE combination op");
          case CombinationOp::EQ:
            throw std::runtime_error("Gradient of sum of equalities not supported");
          default:
            throw std::runtime_error("Failed to recognize combination op during differentiation");
        }
      }
    }
    auto input = op->srcs[idx];
    dop->sink_idxs = std::make_shared<IndexMapExpr>(nullptr, input->idxs);
    dop->sink_dims = std::make_shared<SizeMapExpr>(input->ref->shape.dims_as_exprs());
    dop->ComputeShape(input->ref->shape.layout);
    return dop;
  }

  ExprPtr DeriveOverride(const ExprPtr& dout, const std::shared_ptr<GradOverrideExpr>& op, size_t idx) {
    // TODO: Ideally we'd cache this call somehow so when the only difference is `idx` we don't recompute
    return op->fn->fn(op->out, dout, op->ins, op->fn->user_fn, op->fn->user_ctx)[idx];
  }

  ExprPtr DeriveExtreme(const ExprPtr& dout, const std::shared_ptr<ContractionExpr>& op, size_t idx) {
    // Given `O(oidxs) >= I(iidxs);` (or a MIN aggregation too), produce the derivative
    //  ```dI(iidxs) += (I(iidxs) == O(oidxs)) ? dO(oidxs);```
    // where the above notation is meant to represent a COND combination op
    IVLOG(4, "Gradient::DeriveExtreme> dout=" << dout << ", op=" << op << ", idx=" << idx);
    auto input = op->srcs[0];
    auto dop = std::make_shared<ContractionExpr>();
    dop->agg_op = AggregationOp::SUM;
    dop->combo_op = CombinationOp::COND;
    dop->constraints = op->constraints;
    // Anywhere the forward pass hits the default, the derivative w.r.t. any other tensor is 0;
    // thus, for the corresponding gradient, the default is everywhere zero i.e. the standard unspecified default
    dop->srcs.push_back(input);
    dop->srcs.push_back(std::make_shared<IndexMapExpr>(op, op->sink_idxs->idxs));
    dop->srcs.push_back(std::make_shared<IndexMapExpr>(dout, op->sink_idxs->idxs));
    dop->sink_idxs = std::make_shared<IndexMapExpr>(nullptr, input->idxs);
    dop->sink_dims = std::make_shared<SizeMapExpr>(input->ref->shape.dims_as_exprs());
    dop->ComputeShape(input->ref->shape.layout);
    return dop;
  }

 private:
  ComputeUses uses_;
  std::map<const Expr*, ExprPtr> seen_;
};

}  // namespace

std::vector<ExprPtr> ComputeGradients(const std::vector<ExprPtr>& wrts, const ExprPtr& loss) {
  ExprPtr value = loss;
  auto ndims = loss->shape.dims.size();
  if (ndims) {
    auto cion = std::make_shared<ContractionExpr>();
    cion->agg_op = AggregationOp::SUM;
    cion->combo_op = CombinationOp::NONE;
    std::vector<PolyExprPtr> idxs;
    for (size_t i = 0; i < ndims; i++) {
      idxs.push_back(std::make_shared<PolyIndex>(i));
    }
    cion->srcs = {std::make_shared<IndexMapExpr>(loss, idxs)};
    cion->sink_idxs = std::make_shared<IndexMapExpr>(nullptr, std::vector<PolyExprPtr>{});
    cion->sink_dims = std::make_shared<SizeMapExpr>(std::vector<DimExprPtr>{});
    cion->ComputeShape("");
    value = cion;
  }
  Gradient grad(value);
  std::vector<ExprPtr> ret(wrts.size());
  for (size_t i = 0; i < wrts.size(); i++) {
    ret[i] = grad.GetDerivative(wrts[i]);
  }
  return ret;
}

}  // namespace ast
}  // namespace lang
}  // namespace tile
}  // namespace vertexai
