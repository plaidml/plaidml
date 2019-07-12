// Copyright 2019 Intel Corporation.

#include "tile/lang/ast/gradient.h"

#include <boost/format.hpp>

#include "tile/lang/ast/traversal.h"

namespace vertexai {
namespace tile {
namespace lang {
namespace ast {

namespace {

class Gradient : AstVisitor<void> {
 public:
  Gradient(const ExprPtr& result, const ExprPtr& loss) {
    IVLOG(4, "Gradient::Gradient> result: " << result << ", loss: " << loss);
    seen_[result.get()] = loss;
    auto flat = FlattenAst({result});
    for (auto it = flat.rbegin(); it != flat.rend(); it++) {
      (*it)->Accept(this);
    }
  }

  ExprPtr GetDerivative(const Expr* expr) {
    IVLOG(4, "Gradient::GetDerivative> " << expr);
    auto it = seen_.find(expr);
    if (it == seen_.end()) {
      IVLOG(5, "  not found: " << expr);
      throw std::runtime_error(str(boost::format("Derivative not found: %1%") % expr));
    }
    IVLOG(5, "  returning: " << it->second);
    return it->second;
  }

 private:
  void Visit(const CallExpr& expr) final {
    IVLOG(4, "Gradient::Visit(CallExpr)> " << &expr);
    auto dout = GetDerivative(&expr);
    for (size_t i = 0; i < expr.args.size(); i++) {
      auto dop = CallOp(dout, expr, i);
      AddValue(expr.args[i], dop);
    }
  }

  void Visit(const ContractionExpr& expr) final {
    IVLOG(4, "Gradient::Visit(ContractionExpr)> " << &expr);
    auto dout = GetDerivative(&expr);
    for (size_t i = 0; i < expr.inputs.size(); i++) {
      auto dop = ContractionOp(dout, expr, i);
      AddValue(expr.inputs[i]->ref, dop);
    }
  }

  void Visit(const DimExprExpr& expr) final { IVLOG(4, "Gradient::Visit(DimExprExpr)> " << &expr); }
  void Visit(const FloatConst& expr) final { IVLOG(4, "Gradient::Visit(FloatConst)> " << &expr); }
  void Visit(const IntConst& expr) final { IVLOG(4, "Gradient::Visit(IntConst)> " << &expr); }
  void Visit(const ParamExpr& expr) final { IVLOG(4, "Gradient::Visit(ParamExpr)> " << &expr); }

 private:
  void AddValue(const ExprPtr& expr, const ExprPtr& dop) {
    bool is_new;
    std::map<const Expr*, ExprPtr>::iterator it;
    std::tie(it, is_new) = seen_.emplace(expr.get(), dop);
    if (!is_new) {
      it->second = MakeCall("add", {it->second, dop});
    }
    IVLOG(2, "Gradient::AddValue> expr: " << expr->str() << ", add: " << it->second->str());
    if (it->second->shape.dims.size()) {
      it->second = MakeCall("simple_reduce", {it->second, expr});
    }
  }

  ExprPtr ContractionOp(const ExprPtr& dout, const ContractionExpr& expr, size_t idx) {
    if (expr.use_default && idx == expr.inputs.size() - 1) {
      return DefaultOp(dout, expr);
    }
    if (expr.combo_op == CombinationOp::EQ) {
      return std::make_shared<IntConst>(0);
    }
    if (expr.agg_op == AggregationOp::SUM || expr.agg_op == AggregationOp::ASSIGN) {
      return SumOp(dout, expr, idx);
    }
    if (expr.agg_op == AggregationOp::MIN || expr.agg_op == AggregationOp::MAX) {
      return ExtremeOp(dout, expr, idx);
    }
    if (expr.agg_op == AggregationOp::PROD) {
      throw std::runtime_error("PROD AggregationOp does not support differentiation");
    }
    throw std::runtime_error("Invalid ContractionExpr in ContractionOp");
  }

  ExprPtr CallOp(const ExprPtr& dout, const CallExpr& op, size_t idx) {
    IVLOG(4, "Gradient::CallOp> dout=" << dout << ", op=" << &op << ", fn=" << op.fn << ", idx=" << idx);

    if (op.fn == "tuple") {
      // TODO
      throw std::runtime_error("Not implemented: tuple in CallOp");
      // return FunctionValue::make("element", {dout, IConstValue::make(idx)});
    }

    if (op.fn == "element") {
      // TODO
      throw std::runtime_error("Not implemented: element in CallOp");
      // if (idx == 1) {
      //   return IConstValue::make(0);
      // }
      // const FunctionValue* tuple = dynamic_cast<const FunctionValue*>(op->inputs()[0].get());
      // int64_t elem = dynamic_cast<const IConstValue*>(op->inputs()[1].get())->value();
      // std::vector<ValuePtr> inputs;
      // for (size_t i = 0; i < tuple->inputs().size(); i++) {
      //   if (i == elem) {
      //     inputs.push_back(dout);
      //   } else {
      //     inputs.push_back(IConstValue::make(0));
      //   }
      // }
      // return FunctionValue::make("tuple", inputs);
    }

    if (op.fn == "reshape") {
      // TODO
      throw std::runtime_error("Not implemented: reshape in CallOp");
      // std::vector<ValuePtr> inputs = {dout};
      // ValuePtr in = op->inputs()[0];
      // for (size_t i = 0; i < in->num_dims(); i++) {
      //   inputs.push_back(in->dim_value(i));
      // }
      // return FunctionValue::make("reshape", inputs);
    }

    auto deriv = DerivRegistry::Instance()->Resolve(op.fn);
    auto ptr = std::const_pointer_cast<Expr>(op.as_ptr());
    return deriv.fn(ptr, dout, op.args, deriv.user_fn, deriv.user_ctx)[idx];
  }

  ExprPtr SumOp(const ExprPtr& dout, const ContractionExpr& op, size_t idx) {
    IVLOG(4, "Gradient::SumOp> dout=" << dout << ", op=" << &op << ", idx=" << idx);
    auto dop = std::make_shared<ContractionExpr>();
    dop->agg_op = AggregationOp::SUM;
    dop->combo_op = CombinationOp::NONE;  // May be overridden below based on op->combo_op
    dop->constraints = op.constraints;
    // Anywhere the forward pass hits the default, the derivative w.r.t. any other tensor is 0;
    // thus, for the corresponding gradient, the default is everywhere zero i.e. the standard unspecified default
    for (size_t i = 0; i < op.logical_input_size(); ++i) {
      if (idx == i) {
        dop->inputs.push_back(std::make_shared<TensorSpecExpr>(dout, op.output->index_spec));
      } else {
        switch (op.combo_op) {
          case CombinationOp::MULTIPLY:
            // For *, we multiply by the other (non-differentiated) input
            dop->inputs.push_back(op.inputs[i]);
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
    auto input = op.inputs[idx];
    dop->output = std::make_shared<TensorSpecExpr>(input->index_spec, input->ref->shape.dims_as_exprs());
    dop->ComputeShape(input->ref->shape.layout);
    return dop;
  }

  ExprPtr ExtremeOp(const ExprPtr& dout, const ContractionExpr& op, size_t idx) {
    // Given `O(oidxs) >= I(iidxs);` (or a MIN aggregation too), produce the derivative
    //  ```dI(iidxs) += (I(iidxs) == O(oidxs)) ? dO(oidxs);```
    // where the above notation is meant to represent a COND combination op
    IVLOG(4, "Gradient::ExtremeOp> dout=" << dout << ", op=" << &op << ", idx=" << idx);
    auto input = op.inputs[0];
    auto dop = std::make_shared<ContractionExpr>();
    dop->agg_op = AggregationOp::SUM;
    dop->combo_op = CombinationOp::COND;
    dop->constraints = op.constraints;
    // Anywhere the forward pass hits the default, the derivative w.r.t. any other tensor is 0;
    // thus, for the corresponding gradient, the default is everywhere zero i.e. the standard unspecified default
    dop->inputs.push_back(input);
    auto ptr = std::const_pointer_cast<Expr>(op.as_ptr());
    dop->inputs.push_back(std::make_shared<TensorSpecExpr>(ptr, op.output->index_spec));
    dop->inputs.push_back(std::make_shared<TensorSpecExpr>(dout, op.output->index_spec));
    dop->output = std::make_shared<TensorSpecExpr>(input->index_spec, input->ref->shape.dims_as_exprs());
    dop->ComputeShape(input->ref->shape.layout);
    return dop;
  }

  ExprPtr DefaultOp(const ExprPtr& dout, const ContractionExpr& op) {
    IVLOG(4, "Gradient::DefaultOp> dout=" << dout << ", op=" << &op);
    return dout;
  }

 private:
  std::map<const Expr*, ExprPtr> seen_;
};

}  // namespace

std::vector<ExprPtr> ComputeGradients(  //
    const std::vector<ExprPtr>& wrts,   //
    const ExprPtr& result,              //
    const ExprPtr& loss) {
  ExprPtr value = result;
  // auto ndims = result->shape.dims.size();
  // if (ndims) {
  //   auto cion = std::make_shared<ContractionExpr>();
  //   cion->agg_op = AggregationOp::SUM;
  //   cion->combo_op = CombinationOp::NONE;
  //   std::vector<PolyExprPtr> idxs;
  //   for (size_t i = 0; i < ndims; i++) {
  //     idxs.push_back(std::make_shared<PolyIndex>(i));
  //   }
  //   cion->inputs = {std::make_shared<TensorSpecExpr>(result, idxs)};
  //   cion->output = std::make_shared<TensorSpecExpr>(std::vector<PolyExprPtr>{}, std::vector<DimExprPtr>{});
  //   cion->ComputeShape("");
  //   value = cion;
  // }
  Gradient grad(value, loss);
  std::vector<ExprPtr> ret(wrts.size());
  for (size_t i = 0; i < wrts.size(); i++) {
    ret[i] = grad.GetDerivative(wrts[i].get());
  }
  return ret;
}

}  // namespace ast
}  // namespace lang
}  // namespace tile
}  // namespace vertexai
