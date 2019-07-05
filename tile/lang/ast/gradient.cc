// Copyright 2019 Intel Corporation.

#include "tile/lang/ast/gradient.h"

namespace vertexai {
namespace tile {
namespace lang {
namespace ast {

std::ostream& operator<<(std::ostream& os, const ExprPtr& expr) {
  os << expr->str() << ": " << static_cast<const void*>(expr.get());
  return os;
}

namespace {

// TODO: merge with AstTraversal so that we aren't recursively walking the tree
//       (which could cause the stack to overflow with large networks)
class ComputeUses {
  struct UseInfo {
    ExprPtr ref;
    size_t idx;
  };

 public:
  ComputeUses() = default;

  void Apply(const ExprPtr& expr) {
    if (seen_.count(expr)) {
      return;
    }
    // For Exprs with inputs, walk up the tree
    auto call_expr = std::dynamic_pointer_cast<CallExpr>(expr);
    auto cion_expr = std::dynamic_pointer_cast<ContractionExpr>(expr);
    if (call_expr) {
      for (size_t i = 0; i < call_expr->args.size(); ++i) {
        ExprPtr in = call_expr->args[i];
        uses_[in].emplace_back(UseInfo{call_expr, i});
        Apply(in);
      }
    } else if (cion_expr) {
      for (size_t i = 0; i < cion_expr->inputs.size(); ++i) {
        ExprPtr in = cion_expr->inputs[i]->ref;
        uses_[in].emplace_back(UseInfo{cion_expr, i});
        Apply(in);
      }
    }
    // TODO: There shouldn't be any cycles. Even so, I'm a bit worried about how this works if cycles exist
    seen_.insert(expr);
  }

  const std::vector<UseInfo>& uses(const ExprPtr& expr) {  //
    return uses_[expr];
  }

 private:
  std::map<ExprPtr, std::vector<UseInfo>> uses_;
  std::set<ExprPtr> seen_;
};

class Gradient {
 public:
  Gradient(const ExprPtr& result, const ExprPtr& loss) {
    IVLOG(5, "Gradient::Gradient> result: " << result << ", loss: " << loss);
    uses_.Apply(result);
    seen_[result] = loss;
  }

  ExprPtr ComputeSink(const ExprPtr& expr) {
    IVLOG(4, "Gradient::ComputeSink, expr: " << expr.get());
    auto it = seen_.find(expr);
    if (it != seen_.end()) {
      IVLOG(5, "  Gradient::ComputeSink, already found -> " << it->second);
      return it->second;
    }
    ExprPtr tot;
    for (const auto& use : uses_.uses(expr)) {
      auto dop = OpGrad(ComputeSink(use.ref), use.ref, use.idx);
      if (!tot) {
        tot = dop;
      } else {
        tot = std::make_shared<CallExpr>("add", std::vector<ExprPtr>{tot, dop});
      }
    }
    if (tot == 0) {
      tot = std::make_shared<FloatConst, double>(0.0);
    } else if (true /* tot->num_dims() > 0 */) {  // TODO: this condition isn't set up right
      // TODO: Typically need to perform a simple_reduce to "unbroadcast" here.
      // Currently skipping this to get something running, but this will break for most real networks
      /*
      std::vector<ExprPtr> inputs = {tot};
      for (size_t i = 0; i < expr->num_dims(); ++i) {
        inputs.push_back(expr->dim_value(i));
      }
      // TODO: the `simple_reduce` processing code needs to get ported out of `type.cc`'s TypeCheck
      tot = FunctionValue::make("simple_reduce", inputs);
      */
    }
    IVLOG(5, "  Gradient::ComputeSink, final result -> " << tot);
    seen_.emplace(expr, tot);
    IVLOG(6, "  Result emplaced in `seen_` with expr = " << expr);
    return tot;
  }

 private:
  // TODO: use an AstVisitor to walk the tree (avoiding dynamic casts)
  ExprPtr OpGrad(const ExprPtr& dout, const ExprPtr& op, size_t idx) {
    auto call_op = std::dynamic_pointer_cast<CallExpr>(op);
    auto contraction_op = std::dynamic_pointer_cast<ContractionExpr>(op);
    if (call_op) {
      IVLOG(6, "  Found CallExpr, calling FuncOp");
      return FuncOp(dout, call_op, idx);
    } else if (contraction_op) {
      IVLOG(6, "  Found ContractionExpr");
      if (contraction_op->use_default && idx == contraction_op->inputs.size() - 1) {
        IVLOG(6, "  Calling DefaultOp");
        return DefaultOp(dout, contraction_op);
      } else if (contraction_op->combo_op == CombinationOp::EQ) {
        IVLOG(6, "  Constructing IntConst");
        return std::make_shared<IntConst, int64_t>(0);
      } else if (contraction_op->agg_op == AggregationOp::SUM ||  //
                 contraction_op->agg_op == AggregationOp::ASSIGN) {
        IVLOG(6, "  Calling SumOp");
        return SumOp(dout, contraction_op, idx);
      } else if (contraction_op->agg_op == AggregationOp::MAX ||  //
                 contraction_op->agg_op == AggregationOp::MIN) {
        if (contraction_op->logical_input_size() == 1) {
          IVLOG(6, "  Calling ExtremeOp");
          return ExtremeOp(dout, contraction_op, idx);
        } else {
          throw std::runtime_error("Cannot compute derivative max/min contraction op with more than one input");
        }
      } else if (contraction_op->agg_op == AggregationOp::PROD) {
        throw std::runtime_error("PROD AggregationOp does not support differentiation");
      }
      throw std::runtime_error("Unable to parse ContractionExpr in OpGrad");
    }
    throw std::runtime_error("Invalid operation type in OpGrad");
  }

  ExprPtr FuncOp(const ExprPtr& dout, const std::shared_ptr<CallExpr>& op, size_t idx) {
    IVLOG(4, "  Gradient::FuncOp(), dout=" << dout << ", op=" << op << ", fn=" << op->fn << ", idx=" << idx);

    // TODO : Redo this chunk of specials
    /*
    if (op->fn() == "tuple") {
      return FunctionValue::make("element", {dout, IConstValue::make(idx)});
    }
    if (op->fn() == "element") {
      if (idx == 1) {
        return IConstValue::make(0);
      }
      const FunctionValue* tuple = dynamic_cast<const FunctionValue*>(op->inputs()[0].get());
      int64_t elem = dynamic_cast<const IConstValue*>(op->inputs()[1].get())->value();
      std::vector<ValuePtr> inputs;
      for (size_t i = 0; i < tuple->inputs().size(); i++) {
        if (i == elem) {
          inputs.push_back(dout);
        } else {
          inputs.push_back(IConstValue::make(0));
        }
      }
      return FunctionValue::make("tuple", inputs);
    }
    if (op->fn() == "reshape") {
      std::vector<ValuePtr> inputs = {dout};
      ValuePtr in = op->inputs()[0];
      for (size_t i = 0; i < in->num_dims(); i++) {
        inputs.push_back(in->dim_value(i));
      }
      return FunctionValue::make("reshape", inputs);
    }
    */
    // /TODO : end of chunk of specials

    auto deriv = DerivRegistry::Instance()->Resolve(op->fn);
    return deriv.fn(dout, op, op->args, deriv.user_fn, deriv.user_ctx)[idx];
  }

  ExprPtr SumOp(const ExprPtr& dout, const std::shared_ptr<ContractionExpr>& op, size_t idx) {
    IVLOG(4, "  Gradient::SumOp(), dout=" << dout << ", op=" << op << ", idx=" << idx);
    auto dop = std::make_shared<ContractionExpr>();  // TODO: Mimic this elsewhere
    dop->agg_op = AggregationOp::SUM;
    dop->combo_op = CombinationOp::NONE;  // May be overridden below based on op->combo_op
    dop->constraints = op->constraints;
    // Anywhere the forward pass hits the default, the derivative w.r.t. any other tensor is 0;
    // thus, for the corresponding gradient, the default is everywhere zero i.e. the standard unspecified default
    for (size_t i = 0; i < op->logical_input_size(); ++i) {
      if (idx == i) {
        dop->inputs.push_back(std::make_shared<TensorSpecExpr>(dout, op->output->index_spec));
      } else {
        switch (op->combo_op) {
          case CombinationOp::MULTIPLY:
            // For *, we multiply by the other (non-differentiated) input
            dop->inputs.push_back(op->inputs[i]);
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
    auto input = op->inputs[idx];
    dop->output = std::make_shared<TensorSpecExpr>(input->index_spec, input->ref->shape.dims_as_exprs());
    return dop;
  }

  ExprPtr ExtremeOp(const ExprPtr& dout, const std::shared_ptr<ContractionExpr>& op, size_t idx) {
    // Given `O(oidxs) >= I(iidxs);` (or a MIN aggregation too), produce the derivative
    //  ```dI(iidxs) += (I(iidxs) == O(oidxs)) ? dO(oidxs);```
    // where the above notation is meant to represent a COND combination op
    IVLOG(4, "  Gradient::ExtremeOp(), dout=" << dout << ", op=" << op << ", idx=" << idx);
    auto dop = std::make_shared<ContractionExpr>();
    dop->agg_op = AggregationOp::SUM;
    dop->combo_op = CombinationOp::COND;
    dop->constraints = op->constraints;
    // Anywhere the forward pass hits the default, the derivative w.r.t. any other tensor is 0;
    // thus, for the corresponding gradient, the default is everywhere zero i.e. the standard unspecified default
    dop->inputs.push_back(op->inputs[0]);
    dop->inputs.push_back(std::make_shared<TensorSpecExpr>(op, op->output->index_spec));
    dop->inputs.push_back(std::make_shared<TensorSpecExpr>(dout, op->output->index_spec));
    auto input = op->inputs[0];
    dop->output = std::make_shared<TensorSpecExpr>(input->index_spec, input->ref->shape.dims_as_exprs());
    return dop;
  }

  ExprPtr DefaultOp(const ExprPtr& dout, const std::shared_ptr<ContractionExpr>& op) {
    IVLOG(4, "  Gradient::DefaultOp(), dout=" << dout << ", op=" << op);
    return dout;
  }

 private:
  ComputeUses uses_;
  std::map<ExprPtr, ExprPtr> seen_;
};

}  // namespace

/*
// TODO: This is a draft of an idea for handling `simple_reduce`;
//       I'm uncertain if this kind of function is even the right approach
ExprPtr Gradient::SimpleReduceOp(const ExprPtr& op) {
  std::shared_ptr<ContractionExpr> ret = std::make_shared<ContractionExpr>();
  ret->agg_op = AggregationOp::SUM;
  ret->combo_op = CombinationOp::PLUS;
  // ret->constraints is empty

  // TODO
}
*/

std::vector<ExprPtr> ComputeGradients(  //
    const std::vector<ExprPtr>& wrts,   //
    const ExprPtr& result,              //
    const ExprPtr& loss) {
  Gradient grad(result, loss);
  std::vector<ExprPtr> ret(wrts.size());
  for (size_t i = 0; i < wrts.size(); i++) {
    ret[i] = grad.ComputeSink(wrts[i]);
  }
  return ret;
}

}  // namespace ast
}  // namespace lang
}  // namespace tile
}  // namespace vertexai
