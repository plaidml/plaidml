#include "tile/lang/symbolic.h"

#include <queue>
#include <string>

#include "tile/lang/builtins.h"
#include "tile/lang/parser.h"
#include "tile/lang/sym_poly.h"

namespace vertexai {
namespace tile {
namespace lang {

std::map<ValuePtr, std::set<ValuePtr>> g_deriv_source;

void ComputeUses::Apply(const ValuePtr& val) {
  if (done_.count(val)) {
    return;
  }
  ValueVisitor<void>::Apply(val);
  done_.insert(val);
}

void ComputeUses::Visit(const std::shared_ptr<FunctionValue>& val) {
  for (size_t i = 0; i < val->inputs().size(); i++) {
    ValuePtr in = val->inputs()[i];
    uses_[in].push_back({val, i});
    Apply(in);
  }
}

void ComputeUses::Visit(const std::shared_ptr<ContractionValue>& val) {
  for (size_t i = 0; i < val->inputs().size(); i++) {
    ValuePtr in = val->inputs()[i];
    uses_[in].push_back({val, i});
    Apply(in);
  }
}

Gradient::Gradient() {}

Gradient::Gradient(const ValuePtr& err) : uses_(err) { done_[err] = FConstValue::make(1.0); }

void Gradient::AddSource(const ValuePtr& wrt, const ValuePtr& val) {
  IVLOG(4, "Gradient::AddSource, source: " << wrt);
  uses_.AddTop(wrt);
  done_[wrt] = val;
}

ValuePtr Gradient::operator()(const ValuePtr& val) {
  IVLOG(4, "Gradient::operator(), val: " << val);
  auto it = done_.find(val);
  if (it != done_.end()) {
    IVLOG(4, "  Gradient::operator(), already found -> " << it->second);
    return it->second;
  }
  ValuePtr tot;
  for (const auto& u : uses_.uses(val)) {
    ValuePtr out = u.use;
    ValuePtr dout = (*this)(out);
    ValuePtr dop = OpGrad(dout, out, u.idx);
    if (!tot) {
      tot = dop;
    } else {
      tot = FunctionValue::make("add", {tot, dop});
    }
  }
  if (tot == 0) {
    tot = FConstValue::make(0.0);
  } else if (tot->num_dims() > 0) {
    std::vector<std::shared_ptr<Value>> inputs = {tot};
    for (size_t i = 0; i < val->num_dims(); ++i) {
      inputs.push_back(val->dim_value(i));
    }
    tot = FunctionValue::make("simple_reduce", inputs);
    // g_deriv_source[tot].emplace(val);
    // IVLOG(1, "Saving grad of " << val << " as " << tot);
  }
  IVLOG(4, "  Gradient::operator(), final result -> " << tot);
  done_.emplace(val, tot);
  return tot;
}

ValuePtr Gradient::OpGrad(const ValuePtr& dout, const ValuePtr& op, size_t idx) {
  if (op->type() == Value::Type::FUNCTION) {
    return FuncOp(dout, std::static_pointer_cast<FunctionValue>(op), idx);
  } else if (op->type() == Value::Type::CONTRACTION) {
    auto c = std::static_pointer_cast<ContractionValue>(op);
    if (c->use_default() && idx == c->inputs().size() - 1) {
      return DefaultOp(dout, c);
    } else if (c->comb_op() == CombinationOp::EQ) {
      return IConstValue::make(0);
    } else if (c->agg_op() == AggregationOp::SUM || c->agg_op() == AggregationOp::ASSIGN) {
      return SumOp(dout, c, idx);
    } else if ((c->agg_op() == AggregationOp::MAX || c->agg_op() == AggregationOp::MIN) &&
               c->logical_input_size() == 1) {
      return ExtremeOp(dout, c, idx);
    } else if (c->agg_op() == AggregationOp::PROD) {
      throw std::runtime_error("PROD AggregationOp does not support derivatives yet");
    }
    throw std::runtime_error("Cannot compute derivative max/min contraction op with more than one input");
  }
  throw std::runtime_error("Invalid operation type in OpGrad");
}

ValuePtr Gradient::FuncOp(const ValuePtr& dout, const std::shared_ptr<FunctionValue>& op, size_t idx) {
  IVLOG(4, "  Gradient::FuncOp(), dout=" << dout << ", op=" << op << ", fn=" << op->fn() << ", idx=" << idx);
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
  auto it = DerivDefines.find(op->fn());
  if (it == DerivDefines.end()) {
    throw std::runtime_error("Invalid derivative: unknown function " + op->fn());
  }
  FunctionApplication app(it->second);
  for (size_t i = 0; i < op->inputs().size(); i++) {
    app.SetInput("X" + std::to_string(1 + i), op->inputs()[i]);
  }
  app.SetInput("Y", op);
  app.SetInput("DY", dout);
  return app.GetOutput("DX" + std::to_string(1 + idx));
}

ValuePtr Gradient::SumOp(const ValuePtr& dout, const std::shared_ptr<ContractionValue>& op, size_t idx) {
  IVLOG(4, "  Gradient::SumOp(), dout=" << dout << ", op=" << op << ", idx=" << idx);
  std::vector<SymbolicSpec> specs;
  std::vector<std::shared_ptr<Value>> inputs;
  std::vector<std::shared_ptr<Value>> dims;
  // Set output spec to input idx
  specs.push_back(op->specs()[idx + 1]);
  // Copy the output size from the inputs size
  for (size_t i = 0; i < op->inputs()[idx]->num_dims(); i++) {
    dims.push_back(op->inputs()[idx]->dim_value(i));
  }
  // Go over all the original inputs
  for (size_t i = 0; i < op->specs().size() - 1; i++) {
    if (i == idx) {
      // Put output spec in for the matching index
      specs.push_back(op->specs()[0]);
      inputs.push_back(dout);
    } else if (op->comb_op() == CombinationOp::MULTIPLY) {
      // For multiply, we keep other inputs, for sum, we drop
      specs.push_back(op->specs()[i + 1]);
      inputs.push_back(op->inputs()[i]);
    }
  }
  // Return the result
  return ContractionValue::make(CombinationOp::MULTIPLY, AggregationOp::SUM, specs, op->constraints(), inputs, dims,
                                false, op->no_defract());
}

ValuePtr Gradient::ExtremeOp(const ValuePtr& dout, const std::shared_ptr<ContractionValue>& op, size_t idx) {
  IVLOG(4, "  Gradient::ExtremeOp(), dout=" << dout << ", op=" << op << ", idx=" << idx);
  std::vector<SymbolicSpec> specs = {op->specs()[1], op->specs()[1], op->specs()[0], op->specs()[0]};
  // Get the size of the input (which will be the output for gradient)
  std::vector<std::shared_ptr<Value>> dims;
  for (size_t i = 0; i < op->inputs()[0]->num_dims(); i++) {
    dims.push_back(op->inputs()[0]->dim_value(i));
  }
  ValuePtr out = ContractionValue::make(CombinationOp::COND, AggregationOp::SUM, specs, op->constraints(),
                                        {op->inputs()[0], op, dout}, dims, false, false);
  return out;
}

ValuePtr Gradient::DefaultOp(const ValuePtr& dout, const std::shared_ptr<ContractionValue>& op) {
  IVLOG(4, "  Gradient::DefaultOp(), dout=" << dout << ", op=" << op);
  return dout;
}

Program ProgGrad(const Program& p) {
  auto bf = std::make_shared<BoundFunction>(p, std::vector<std::shared_ptr<TensorValue>>{});
  FunctionApplication fa(bf);
  BoundFunction newbf;
  IVLOG(4, "Making derivative of " << to_string(p));
  // Make placeholders for each input + add to application + new bound function
  std::map<std::string, std::shared_ptr<PlaceholderValue>> new_ins;
  for (const auto& in : p.inputs) {
    if (in.tag == Input::VARIABLE) {
      throw std::runtime_error("Invalid variable sized input in ProgGrad");
    }
    auto pv = std::make_shared<PlaceholderValue>(in.dims.size());
    new_ins[in.name] = pv;
    fa.SetInput(in.name, pv);
    newbf.AddInput(in.name, pv);
  }
  // Make a gradient thingy and add gradient input values to the new function
  Gradient g;
  for (const auto& out : p.outputs) {
    ValuePtr ov = fa.GetOutput(out);
    auto pv = std::make_shared<PlaceholderValue>(ov->num_dims());
    g.AddSource(ov, pv);
    newbf.AddInput("_d_" + out, pv);
  }
  // Add the gradient outputs
  for (const auto& in : p.inputs) {
    newbf.AddOutput("_d_" + in.name, g(new_ins[in.name]));
  }
  newbf.Done();

  return Xify(newbf.prog());
}

}  // namespace lang
}  // namespace tile
}  // namespace vertexai
