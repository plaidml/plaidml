#include "tile/lang/compose.h"

#include <algorithm>

#include "tile/lang/builtins.h"
#include "tile/lang/fpconv.h"
#include "tile/lang/gen_special.h"
#include "tile/lang/replace.h"
#include "tile/lang/sym_poly.h"
#include "tile/lang/symbolic.h"
#include "tile/lang/type.h"

namespace vertexai {
namespace tile {
namespace lang {

using std::static_pointer_cast;
using std::to_string;

std::shared_ptr<IConstValue> IConstValue::make(const int64_t& val) {
  auto result = Interned<IConstValue>::make(val);
  IVLOG(4, "Making IConstValue " << result.get() << " from constant " << val);
  return result;
}

std::map<std::shared_ptr<Value>, std::set<std::string>> g_ids;

std::shared_ptr<Value> FunctionValue::make(std::string fn, std::vector<std::shared_ptr<Value>> inputs) {
  static std::shared_ptr<Value> zeroi = IConstValue::make(0);
  static std::shared_ptr<Value> onei = IConstValue::make(1);
  static std::shared_ptr<Value> zerof = FConstValue::make(0);
  static std::shared_ptr<Value> onef = FConstValue::make(1);

  if (fn == "ident") {
    return inputs[0];
  }
  if (fn == "sub") {
    if (inputs[1]->type() == ICONST) {
      if (inputs[0]->type() == ICONST) {
        return IConstValue::make(dynamic_cast<const IConstValue*>(inputs[0].get())->value() -
                                 dynamic_cast<const IConstValue*>(inputs[1].get())->value());
      }
      if (inputs[0]->type() == FCONST) {
        return FConstValue::make(dynamic_cast<const FConstValue*>(inputs[0].get())->value() -
                                 dynamic_cast<const IConstValue*>(inputs[1].get())->value());
      }
      // Rewrite into an add with the constant on the lhs, slightly simplifying subsequent merge logic.
      fn = "add";
      inputs[1] = IConstValue::make(0 - dynamic_cast<const IConstValue*>(inputs[1].get())->value());
      std::swap(inputs[0], inputs[1]);
    }
    if (inputs[1]->type() == FCONST) {
      if (inputs[0]->type() == ICONST) {
        return FConstValue::make(dynamic_cast<const IConstValue*>(inputs[0].get())->value() -
                                 dynamic_cast<const FConstValue*>(inputs[1].get())->value());
      }
      if (inputs[0]->type() == FCONST) {
        return FConstValue::make(dynamic_cast<const FConstValue*>(inputs[0].get())->value() -
                                 dynamic_cast<const FConstValue*>(inputs[1].get())->value());
      }
      // Rewrite into an add with the constant on the lhs, slightly simplifying subsequent merge logic.
      fn = "add";
      inputs[1] = FConstValue::make(0.0 - dynamic_cast<const FConstValue*>(inputs[1].get())->value());
      std::swap(inputs[0], inputs[1]);
    }
  }
  if (fn == "add") {
    if (inputs[0] == zeroi || inputs[0] == zerof) {
      return inputs[1];
    }
    if (inputs[1] == zeroi || inputs[1] == zerof) {
      return inputs[0];
    }
    if (inputs[1]->type() == ICONST) {
      if (inputs[0]->type() == ICONST) {
        return IConstValue::make(dynamic_cast<const IConstValue*>(inputs[0].get())->value() +
                                 dynamic_cast<const IConstValue*>(inputs[1].get())->value());
      }
      if (inputs[0]->type() == FCONST) {
        return FConstValue::make(dynamic_cast<const FConstValue*>(inputs[0].get())->value() +
                                 dynamic_cast<const IConstValue*>(inputs[1].get())->value());
      }
      // Rewrite to put the constant on the lhs, slightly simplifying subsequent merge logic.
      std::swap(inputs[0], inputs[1]);
    }
    if (inputs[1]->type() == FCONST) {
      if (inputs[0]->type() == ICONST) {
        return FConstValue::make(dynamic_cast<const IConstValue*>(inputs[0].get())->value() +
                                 dynamic_cast<const FConstValue*>(inputs[1].get())->value());
      }
      if (inputs[0]->type() == FCONST) {
        return FConstValue::make(dynamic_cast<const FConstValue*>(inputs[0].get())->value() +
                                 dynamic_cast<const FConstValue*>(inputs[1].get())->value());
      }
      // Rewrite to put the constant on the lhs, slightly simplifying subsequent merge logic.
      std::swap(inputs[0], inputs[1]);
    }
    if (inputs[1]->type() == FUNCTION && (inputs[0]->type() == ICONST || inputs[0]->type() == FCONST)) {
      const auto* lhs = dynamic_cast<FunctionValue*>(inputs[1].get());
      if (lhs->fn() == "add" || lhs->fn() == "sub") {
        if (lhs->inputs_[0]->type() == ICONST) {
          if (inputs[0]->type() == ICONST) {
            return FunctionValue::make(
                lhs->fn(), {IConstValue::make(dynamic_cast<const IConstValue*>(lhs->inputs_[0].get())->value() +
                                              dynamic_cast<const IConstValue*>(inputs[0].get())->value()),
                            lhs->inputs_[1]});
          }
          // inputs[0]->type() == FCONST
          return FunctionValue::make(
              lhs->fn(), {FConstValue::make(dynamic_cast<const IConstValue*>(lhs->inputs_[0].get())->value() +
                                            dynamic_cast<const FConstValue*>(inputs[0].get())->value()),
                          lhs->inputs_[1]});
        }
        if (lhs->inputs_[0]->type() == FCONST) {
          if (inputs[0]->type() == ICONST) {
            return FunctionValue::make(
                lhs->fn(), {FConstValue::make(dynamic_cast<const FConstValue*>(lhs->inputs_[0].get())->value() +
                                              dynamic_cast<const IConstValue*>(inputs[0].get())->value()),
                            lhs->inputs_[1]});
          }
          // inputs[0]->type() == FCONST
          return FunctionValue::make(
              "add", {FConstValue::make(dynamic_cast<const FConstValue*>(lhs->inputs_[0].get())->value() +
                                        dynamic_cast<const FConstValue*>(inputs[0].get())->value()),
                      lhs->inputs_[1]});
        }
      }
    }
  }
  if (fn == "div") {
    if (inputs[1] == onei || inputs[1] == onef) {
      return inputs[0];
    }
  }
  if (fn == "mul") {
    if (inputs[0] == zeroi || inputs[0] == zerof) {
      return inputs[0];
    }
    if (inputs[1] == zeroi || inputs[1] == zerof) {
      return inputs[1];
    }
    if (inputs[0] == onei || inputs[0] == onef) {
      return inputs[1];
    }
    if (inputs[1] == onei || inputs[1] == onef) {
      return inputs[0];
    }
  }
  if (fn == "match" && inputs[0] == inputs[1]) {
    return inputs[0];
  }
  if (fn == "cond" && inputs[1]->num_dims() == inputs[2]->num_dims()) {
    if (inputs[0]->type() == Value::FCONST) {
      return std::dynamic_pointer_cast<FConstValue>(inputs[0])->value() ? inputs[1] : inputs[2];
    } else if (inputs[0]->type() == Value::ICONST) {
      return std::dynamic_pointer_cast<IConstValue>(inputs[0])->value() ? inputs[1] : inputs[2];
    }
  }
  if (fn == "broadcast") {
    if (inputs[1] == onei) {
      return inputs[0];
    }
    if (inputs[0] == onei) {
      return inputs[1];
    }
    if (inputs[0] == inputs[1]) {
      return inputs[0];
    }
  }
  if (fn == "log" && inputs.size() == 1) {
    auto inner = std::dynamic_pointer_cast<FunctionValue>(inputs[0]);
    if (inner && inner->fn() == "builtin_softmax") {
      return FunctionValue::make("builtin_logsoftmax", inner->inputs());
    }
  }

  auto result = Interned<FunctionValue>::make(fn, inputs);
  IVLOG(4, "Making FunctionValue " << *result << " from fn " << fn);
  for (auto in : inputs) {
    IVLOG(4, "  Input " << *in);
    for (size_t i = 0; i < in->num_dims(); ++i) {
      IVLOG(4, "    Dim " << *in->dim_value(i));
    }
  }
  return result;
}

FunctionValue::FunctionValue(std::string fn, std::vector<std::shared_ptr<Value>> inputs)
    : fn_{std::move(fn)}, inputs_{std::move(inputs)} {
  IVLOG(4, "Building function value \"" << fn_ << "\" over " << inputs_.size() << " inputs");
  if (fn_ == "prng_step" || fn_ == "reshape") {
    if (inputs_.size() < 1) {
      throw std::runtime_error("prng_step/reshape must have at least one input");
    }
    for (size_t i = 1; i < inputs_.size(); i++) {
      if (inputs_[i]->num_dims() != 0) {
        throw std::runtime_error("prng_step/reshape sizes must be scalars");
      }
      dims_.push_back(inputs_[i]);
    }
    return;
  }
  if (fn_ == "prng_value") {
    if (inputs_.size() != 1) {
      throw std::runtime_error("prng_value must have exactly one input");
    }
    for (size_t i = 0; i < inputs_[0]->num_dims(); i++) {
      dims_.push_back(inputs_[0]->dim_value(i));
    }
    return;
  }
  if (fn_ == "prng_state") {
    dims_.push_back(IConstValue::make(3));
    dims_.push_back(IConstValue::make(k_rng_size));
    return;
  }
  if (fn_ == "shape") {
    if (inputs_.size() != 1) {
      throw std::runtime_error("shape must have exactly one input");
    }
    dims_.push_back(IConstValue::make(inputs_[0]->num_dims()));
    return;
  }
  // TODO: "gather" + "scatter"
  size_t max_dims = 0;
  for (size_t i = 0; i < inputs_.size(); i++) {
    IVLOG(4, "  input[" << i << "]->num_dims is " << inputs_[i]->num_dims());
    max_dims = std::max(max_dims, inputs_[i]->num_dims());
    IVLOG(4, "  max_dims is now " << max_dims);
  }
  IVLOG(4, "  Final max_dims is " << max_dims);
  dims_.resize(max_dims);
  for (size_t i = 0; i < max_dims; i++) {
    for (size_t j = 0; j < inputs_.size(); j++) {
      auto offset = max_dims - inputs_[j]->num_dims();
      if (i < offset) {
        continue;
      }
      if (!dims_[i]) {
        dims_[i] = inputs_[j]->dim_value(i - offset);
        continue;
      }
      if (dims_[i] != inputs_[j]->dim_value(i - offset)) {
        dims_[i] = FunctionValue::make("broadcast", {dims_[i], inputs_[j]->dim_value(i - offset)});
      }
    }
  }
}

std::shared_ptr<Value> ContractionValue::make(CombinationOp comb_op, AggregationOp agg_op,
                                              const std::vector<SymbolicSpec>& specs,
                                              const std::vector<ValueConstraint>& constraints,
                                              const std::vector<std::shared_ptr<Value>>& inputs,
                                              const std::vector<std::shared_ptr<Value>>& dims, bool no_defract) {
  auto result = std::shared_ptr<Value>(
      Interned<ContractionValue>::make(comb_op, agg_op, specs, constraints, inputs, dims, no_defract));
  IVLOG(4, "Making ContractionValue " << result.get() << " comb_op=" << static_cast<char>(comb_op)
                                      << " agg_op=" << static_cast<char>(agg_op));
  for (auto in : inputs) {
    IVLOG(4, "  Input " << in);
  }
  for (auto dim : dims) {
    IVLOG(4, "  Dim " << dim);
  }
  return result;
}

BoundFunction::BoundFunction(const std::string& code, const std::string& id) {
  Parser p;
  prog_ = p.Parse(code, id);
  for (size_t i = 0; i < prog_.inputs.size(); i++) {
    in_pos_[prog_.inputs[i].name] = i;
  }
  for (size_t i = 0; i < prog_.outputs.size(); i++) {
    out_pos_[prog_.outputs[i]] = i;
  }
}

BoundFunction::BoundFunction(const Program& prog, const std::vector<std::shared_ptr<TensorValue>>& bound_inputs) {
  prog_ = prog;
  if (bound_inputs.size() > prog_.inputs.size()) {
    throw std::runtime_error("Not enough inputs to program in BoundFunction load");
  }
  size_t unbound_inputs = prog_.inputs.size() - bound_inputs.size();
  for (size_t i = 0; i < unbound_inputs; i++) {
    if (prog_.inputs[i].name[0] == '_') {
      throw std::runtime_error("In BoundFunction load, input type mismatch");
    }
    in_pos_[prog_.inputs[i].name] = i;
  }
  for (size_t i = 0; i < bound_inputs.size(); i++) {
    std::string name = prog_.inputs[unbound_inputs + i].name;
    if (name[0] != '_') {
      throw std::runtime_error("In BoundFunction load, input type mismatch 2");
    }
    in_bound_[name] = bound_inputs[i];
  }
  for (size_t i = 0; i < prog_.outputs.size(); i++) {
    out_pos_[prog_.outputs[i]] = i;
  }
}

void BoundFunction::AddInput(const std::string& name, const std::shared_ptr<PlaceholderValue>& val) {
  if (updated_.size()) {
    throw std::runtime_error("Cannot add inputs after updates: " + name);
  }
  if (out_pos_.size()) {
    throw std::runtime_error("Cannot add inputs after outputs: " + name);
  }
  if (in_pos_.count(name)) {
    throw std::runtime_error("Duplicate input name: " + name);
  }
  size_t prev_size = in_pos_.size();
  in_pos_[name] = prev_size;  // Operator[] happens *before* rhs runs
  Input in = {Input::FIXED, name};
  bindings_[val] = name;
  for (size_t j = 0; j < val->num_dims(); j++) {
    std::string dname = "_" + name + "_" + std::to_string(j);
    in.dims.push_back(dname);
    bindings_[val->dim_value(j)] = dname;
  }
  prog_.inputs.push_back(in);
}

void BoundFunction::AddOutput(const std::string& name, const std::shared_ptr<Value>& val) {
  if (updated_.size()) {
    throw std::runtime_error("Cannot add outputs after updates: " + name);
  }
  if (out_pos_.count(name)) {
    throw std::runtime_error("Duplicate output name: " + name);
  }
  out_pos_[name] = out_pos_.size();
  std::string oname = Apply(val);
  // TODO: Should I do this as a replacement?  An issue with this is a direct return of an input
  Op op = {Op::FUNCTION, name, {oname}, {}, {"ident"}};
  prog_.ops.push_back(op);
  prog_.outputs.push_back(name);
}

void BoundFunction::AddDependency(const FunctionApplication& prev) {
  if (!prev.is_done()) {
    throw std::runtime_error("Adding a dependency on an incomplete function application");
  }
  for (const auto& kvp : prev.updates()) {
    AddUpdate(kvp.first, kvp.second);
  }
}

void BoundFunction::AddUpdate(const std::shared_ptr<TensorValue>& lhs, const std::shared_ptr<Value>& rhs) {
  if (updated_.count(lhs)) {
    // TODO: Make new updates override old updates
    throw std::runtime_error("Duplicate updates");
  }
  std::string oname = Apply(rhs);
  if (oname.size() > 2 && oname.substr(0, 2) == "_I") {
    // Handle case where output is a stright copy of an input by inserting
    // an identity function
    std::string tmp = NewTmp();
    Op op = {Op::FUNCTION, tmp, {oname}, {}, {"ident"}};
    prog_.ops.push_back(op);
    oname = tmp;
  }
  out_bound_[oname] = lhs;
  prog_.outputs.push_back(oname);
  updated_.emplace(lhs);
}

void BoundFunction::Done() { bindings_.clear(); }

Program Xify(const Program& orig) {
  Program r = orig;
  for (auto& i : r.inputs) {
    i.name = "X" + i.name;
    for (auto& d : i.dims) {
      d = "X" + d;
    }
  }
  for (auto& o : r.outputs) {
    o = "X" + o;
  }
  for (auto& op : r.ops) {
    op.output = "X" + op.output;
    if (op.tag == Op::CONSTANT) {
      continue;
    }
    for (auto& i : op.inputs) {
      i = "X" + i;
    }
    if (op.tag == Op::FUNCTION) {
      continue;
    }
    for (auto& s : op.c.output_size) {
      s = "X" + s;
    }
    for (auto& s : op.c.specs) {
      s.id = "X" + s.id;
      for (auto& ss : s.sspec) {
        ss = ss->Xify();
      }
    }
    for (auto& c : op.c.constraints) {
      c.range = "X" + c.range;
    }
  }
  return r;
}

static std::string DeX(const std::string& s) {
  if (!s.size() || s[0] != 'X') {
    throw std::runtime_error("Not an X in DeX");
  }
  return s.substr(1, s.size() - 1);
}

Program DeXify(const Program& orig) {
  Program r = orig;
  for (auto& i : r.inputs) {
    i.name = DeX(i.name);
    for (auto& d : i.dims) {
      d = DeX(d);
    }
  }
  for (auto& o : r.outputs) {
    o = DeX(o);
  }
  for (auto& op : r.ops) {
    op.output = DeX(op.output);
    if (op.tag == Op::CONSTANT) {
      continue;
    }
    for (auto& i : op.inputs) {
      i = DeX(i);
    }
    if (op.tag == Op::FUNCTION) {
      continue;
    }
    for (auto& s : op.c.output_size) {
      s = DeX(s);
    }
    for (auto& s : op.c.specs) {
      s.id = DeX(s.id);
      for (auto& ss : s.sspec) {
        ss = ss->DeXify();
      }
    }
    for (auto& c : op.c.constraints) {
      c.range = DeX(c.range);
    }
  }
  return r;
}

RunInfo BoundFunction::PrepareToRun() const {
  if (num_inputs() != 0) {
    throw std::runtime_error("Unable to run function with unbound inputs");
  }
  if (num_outputs() != 0) {
    throw std::runtime_error("Unable to run function with unbound outputs");
  }

  RunInfo r;
  Program inlined = prog_;
  ApplyDefines(&inlined, InlineDefines);
  Program xp = Xify(inlined);

  for (const auto& kvp : in_bound_) {
    std::string n = "X" + kvp.first;
    r.input_shapes[n] = kvp.second->shape();
    r.input_buffers[n] = kvp.second->buffer();
  }
  for (const auto& kvp : out_bound_) {
    std::string n = "X" + kvp.first;
    r.output_shapes[n] = kvp.second->shape();
    r.output_buffers[n] = kvp.second->buffer();
  }

  auto bindings = BindProgram(&xp, r.input_shapes, r.output_shapes);
  for (auto& op : xp.ops) {
    if (op.tag != Op::CONTRACTION) {
      continue;
    }
    for (auto& s : op.c.output_size) {
      if (!(s[0] >= '0' && s[0] <= '9')) {
        s = std::to_string(bindings.at(s).iconst);
      }
    }
    for (auto& c : op.c.constraints) {
      c.range = "";
    }
  }
  r.code = to_string(xp);

  return r;
}

std::string BoundFunction::Apply(const std::shared_ptr<Value>& val) {
  auto it = bindings_.find(val);
  if (it != bindings_.end()) {
    return it->second;
  }
  std::string name = ValueVisitor<std::string>::Apply(val);
  auto it2 = g_ids.find(val);
  if (it2 != g_ids.end()) {
    Attribute attr = {"pid", {}};
    for (const auto& s : it2->second) {
      attr.params.push_back(s);
    }
    prog_.ops.back().attributes.emplace_back(attr);
  }
  /*
  auto it2 = g_deriv_source.find(val);
  if (it2 != g_deriv_source.end()) {
    Attribute d_of = { "d_of", {} };
    for(const auto& s : it2->second) {
      auto it3 = bindings_.find(s);
      if (it3 == bindings_.end()) {
        d_of.params.push_back("unknown");
      } else {
        d_of.params.push_back(it3->second);
      }
    }
    prog_.ops.back().attributes.push_back(d_of);
  }
  */
  bindings_[val] = name;
  return name;
}

std::string BoundFunction::Visit(const std::shared_ptr<TensorValue>& val) {
  std::string tname = "_I_" + std::to_string(in_bound_.size());
  Input in = {Input::FIXED, tname};
  for (size_t i = 0; i < val->num_dims(); i++) {
    std::string dname = tname + "_" + std::to_string(i);
    in.dims.push_back(dname);
    bindings_[val->dim_value(i)] = dname;
  }
  in_bound_.emplace(tname, val);
  prog_.inputs.push_back(in);
  return tname;
}

std::string BoundFunction::Visit(const std::shared_ptr<PlaceholderValue>& val) {
  throw std::runtime_error("Binding missing placeholder");
}

std::string BoundFunction::Visit(const std::shared_ptr<FConstValue>& val) {
  std::string str = DoubleToString(val->value());
  if (str.find_first_of(".e") == std::string::npos) {
    str.append(".0");
  }
  Op op = {Op::CONSTANT, NewTmp(), {str}, {}, {"fconst"}};
  prog_.ops.push_back(op);
  return op.output;
}

std::string BoundFunction::Visit(const std::shared_ptr<IConstValue>& val) {
  Op op = {Op::CONSTANT, NewTmp(), {std::to_string(val->value())}, {}, {"iconst"}};
  IVLOG(4, "Allocating iconst " << op.output);
  prog_.ops.push_back(op);
  return op.output;
}

std::string BoundFunction::Visit(const std::shared_ptr<FunctionValue>& val) {
  std::vector<std::string> inputs;
  for (size_t i = 0; i < val->inputs().size(); i++) {
    inputs.push_back(Apply(val->inputs()[i]));
  }
  Op op = {Op::FUNCTION, NewTmp(), inputs, {}, {val->fn()}};
  IVLOG(4, "Allocated function " << op);
  prog_.ops.push_back(op);
  return op.output;
}

static SymbolicSpec DecomposeSpecs(const SymbolicSpec& orig, BoundFunction* bf) {
  SymbolicSpec out;
  for (size_t i = 0; i < orig.size(); i++) {
    out.push_back(orig[i]->Decompose(bf));
  }
  return out;
}

std::string BoundFunction::Visit(const std::shared_ptr<ContractionValue>& val) {
  Op op = {Op::CONTRACTION, NewTmp()};
  auto& inputs = op.inputs;
  auto& c = op.c;
  c.comb_op = val->comb_op();
  c.agg_op = val->agg_op();
  IVLOG(4, "Building op to produce " << op.output);
  for (size_t i = 0; i < val->num_dims(); i++) {
    std::string dsize = Apply(val->dim_value(i));
    IVLOG(4, "  Pushing dsize=" << dsize);
    c.output_size.push_back(dsize);
  }
  c.specs.push_back(TensorSpec{op.output, DecomposeSpecs(val->specs()[0], this)});
  for (size_t i = 0; i < val->inputs().size(); i++) {
    std::string vname = Apply(val->inputs()[i]);
    inputs.push_back(vname);
    c.specs.push_back(TensorSpec{vname, DecomposeSpecs(val->specs()[i + 1], this)});
  }
  for (const auto& vc : val->constraints()) {
    std::string rname = Apply(vc.range);
    SymbolicConstraint rc(vc.poly->Decompose(this), rname);
    c.constraints.push_back(rc);
  }
  c.no_defract = val->no_defract();
  IVLOG(4, "Built op " << op);
  prog_.ops.push_back(op);
  return op.output;
}

FunctionApplication::FunctionApplication(const std::shared_ptr<BoundFunction>& func)
    : func_{func}, attached_{func_->in_bound().size()} {
  for (const auto& kvp : func_->in_bound()) {
    bindings_.emplace(kvp.first, kvp.second);
    IVLOG(4, "FunApp::FunApp " << this << " binding " << kvp.first << " -> " << *kvp.second);
  }
}

void FunctionApplication::AddDependency(const FunctionApplication& prev) {
  if (!prev.is_done()) {
    throw std::runtime_error("Adding a dependency on an incomplete function application");
  }
  for (auto& kvp : bindings_) {
    auto tp = std::dynamic_pointer_cast<TensorValue>(kvp.second);
    if (!tp) {
      throw std::runtime_error("Add dependencies before setting inputs");
    }
    for (auto& update : prev.updates()) {
      if (update.first == tp) {
        kvp.second = update.second;
        break;
      }
    }
  }
  for (const auto& kvp : prev.updates()) {
    updates_.emplace_back(kvp.first, kvp.second);
  }
}

void FunctionApplication::SetInput(const std::string& name, const std::shared_ptr<Value>& val) {
  if (is_done_) {
    throw std::runtime_error("Attempting to set input after outputs generated");
  }
  if (bindings_.count(name)) {
    throw std::runtime_error("Duplicate input parameter on apply: " + name);
  }
  if (!func_->in_pos().count(name)) {
    throw std::runtime_error("Unknown input parameter on apply: " + name);
  }
  const Program& p = func_->prog();
  const Input& pi = p.inputs[func_->in_pos().at(name)];
  if (pi.tag == Input::FIXED) {
    if (val->num_dims() != pi.dims.size()) {
      throw std::runtime_error("Applying function, tensor with mismatching dimensionality: " + name + ", expected=" +
                               to_string(pi.dims.size()) + ", got=" + to_string(val->num_dims()));
    }
    for (size_t d = 0; d < pi.dims.size(); d++) {
      bindings_[pi.dims[d]] = val->dim_value(d);
      IVLOG(4, "FunApp::SetInput " << this << " binding fixed " << pi.dims[d] << " -> " << *val->dim_value(d));
    }
  }
  bindings_[pi.name] = val;
  IVLOG(4, "FunApp::SetInput " << this << " binding " << pi.name << " -> " << *val);
  attached_++;
}

static SymbolicSpec ComposeSpecs(const SymbolicSpec& orig, const FunctionApplication& fa) {
  SymbolicSpec out;
  for (size_t i = 0; i < orig.size(); i++) {
    out.push_back(orig[i]->Compose(fa));
  }
  return out;
}

void FunctionApplication::SetDone() {
  if (is_done_) {
    return;
  }
  const Program& p = func_->prog();
  if (attached_ != p.inputs.size()) {
    throw std::runtime_error("Missing inputs in apply");
  }

  // Walk over bound inputs
  for (size_t i = func_->in_pos().size(); i < p.inputs.size(); i++) {
    const Input& pi = p.inputs[i];
    const std::shared_ptr<Value> val = func_->in_bound().at(pi.name);
    if (pi.tag == Input::FIXED) {
      if (val->num_dims() != pi.dims.size()) {
        throw std::runtime_error("Applying function, tensor with mismatching dimensionality: " + pi.name);
      }
      for (size_t d = 0; d < pi.dims.size(); d++) {
        bindings_[pi.dims[d]] = val->dim_value(d);
        IVLOG(4, "FunApp::SetDone " << this << " binding " << pi.dims[d] << " -> " << *val->dim_value(d));
      }
    }
  }

  // Walk over all the ops
  for (size_t i = 0; i < p.ops.size(); i++) {
    const Op& o = p.ops[i];
    if (o.tag == Op::CONSTANT) {
      if (o.f.fn == "iconst") {
        bindings_.emplace(o.output, IConstValue::make(int64_t(atol(o.inputs[0].c_str()))));
        IVLOG(4, "FunApp::SetDone " << this << " binding " << o.output << " ->(iconst) " << o.inputs[0]);
      } else {
        bindings_.emplace(o.output, FConstValue::make(atof(o.inputs[0].c_str())));
        IVLOG(4, "FunApp::SetDone " << this << " binding " << o.output << " ->(fconst) " << o.inputs[0]);
      }
    } else if (o.tag == Op::FUNCTION) {
      std::vector<std::shared_ptr<Value>> inputs;
      for (const auto& s : o.inputs) {
        try {
          inputs.push_back(bindings_.at(s));
        } catch (std::out_of_range) {
          throw std::runtime_error(std::string("Missing input binding \"") + s + "\" in function op");
        }
      }
      bindings_[o.output] = FunctionValue::make(o.f.fn, inputs);
      IVLOG(4, "FunApp::SetDone " << this << " binding " << o.output << " ->(func) " << *bindings_[o.output]);
    } else {  // Contraction
      const Contraction& c = o.c;
      std::vector<std::shared_ptr<Value>> inputs;
      std::vector<std::shared_ptr<Value>> dims;
      std::vector<SymbolicSpec> specs;
      std::vector<ValueConstraint> cons;
      IVLOG(4, "Making ContractionValue for op " << o << " in app " << this);
      for (const auto& s : o.inputs) {
        try {
          inputs.push_back(bindings_.at(s));
          IVLOG(4, "  Added input " << s << " -> " << *bindings_.at(s));
        } catch (std::out_of_range) {
          throw std::runtime_error(std::string("Missing input binding \"") + s + "\" in contraction op");
        }
      }
      for (const auto& s : c.output_size) {
        try {
          dims.push_back(bindings_.at(s));
          IVLOG(4, "  Added output dim " << s << " -> " << *bindings_.at(s));
        } catch (std::out_of_range) {
          throw std::runtime_error(std::string("Missing output binding \"") + s + "\"");
        }
      }
      for (const auto& ts : c.specs) {
        specs.push_back(ComposeSpecs(ts.sspec, *this));
      }
      for (const auto& con : c.constraints) {
        std::shared_ptr<Value> top = bindings_.at(con.range);
        cons.push_back(ValueConstraint({con.poly->Compose(*this), top}));
      }
      bool no_defract = c.no_defract;
      bindings_[o.output] = ContractionValue::make(c.comb_op, c.agg_op, specs, cons, inputs, dims, no_defract);
      IVLOG(4, "FunApp::SetDone " << this << " binding " << o.output << " ->(contraction) " << *bindings_[o.output]);
    }
    for (const auto& attr : o.attributes) {
      if (attr.name == "pid") {
        for (const auto& s : attr.params) {
          g_ids[bindings_[o.output]].emplace(s);
        }
      }
    }
  }
  // Run the 'updates'
  for (const auto& kvp : func_->out_bound()) {
    updates_.emplace_back(kvp.second, bindings_[kvp.first]);
  }
  is_done_ = true;
}

std::shared_ptr<Value> FunctionApplication::GetOutput(const std::string& name) {
  SetDone();
  if (!func_->out_pos().count(name)) {
    throw std::runtime_error("Unknown output parameter on apply: " + name);
  }
  IVLOG(4, "FunApp::GetOutput " << this << " " << name << " = " << bindings_.at(name));
  return bindings_.at(name);
}

namespace {
class TypeBindingVisitor final : public ValueVisitor<void> {
 public:
  TypeBindingVisitor(const std::string& name, Bindings* bindings) : name_{name}, bindings_{bindings} {}

  void Visit(const std::shared_ptr<TensorValue>& val) final { bindings_->emplace(name_, Binding(val->shape())); }
  void Visit(const std::shared_ptr<PlaceholderValue>&) final {
    // throw std::runtime_error("Unable to determine concrete shape involving placeholder input \"" + name_ + "\"");
  }
  void Visit(const std::shared_ptr<FConstValue>& val) final { bindings_->emplace(name_, Binding(val->value())); }
  void Visit(const std::shared_ptr<IConstValue>& val) final { bindings_->emplace(name_, Binding(val->value())); }
  void Visit(const std::shared_ptr<FunctionValue>&) final {
    // throw std::runtime_error("Unable to determine concrete shape involving function input \"" + name_ + "\"");
  }
  void Visit(const std::shared_ptr<ContractionValue>&) final {
    // throw std::runtime_error("Unable to determine concrete shape involving contraction input \"" + name_ + "\"");
  }

 private:
  const std::string name_;
  Bindings* bindings_;
};
}  // namespace

TensorShape FunctionApplication::GetOutputShape(const std::string& name) {
  if (!is_done_) {
    SetDone();
  }
  IVLOG(4, "Getting shape of output " << name);
  if (!is_typechecked_) {
    Bindings typecheck_bindings;

    // Copy the program, because typechecking modifies it.
    Program prog = func_->prog();

    for (const auto& input : prog.inputs) {
      auto it = bindings_.find(input.name);
      if (it != bindings_.end()) {
        TypeBindingVisitor(it->first, &typecheck_bindings).Apply(it->second);
      }
    }

    TypeCheck(&prog, &typecheck_bindings);

    typecheck_bindings_.swap(typecheck_bindings);
    is_typechecked_ = true;
  }

  if (!typecheck_bindings_.count(name)) {
    throw std::runtime_error("Unknown output parameter on apply: " + name);
  }

  const Binding& binding = typecheck_bindings_.at(name);
  if (binding.tag != Binding::TENSOR) {
    throw std::runtime_error("Output parameter " + name + " is not a tensor");
  }

  return binding.shape;
}

void Value::log(el::base::type::ostream_t& os) const {
  switch (type()) {
    case Value::Type::TENSOR: {
      // const auto* v = static_cast<const TensorValue*>(this);
      os << "Tensor";
      break;
    }
    case Value::Type::PLACEHOLDER:
      os << "Placeholder";
      break;
    case Value::Type::FCONST: {
      const auto* v = static_cast<const FConstValue*>(this);
      os << "FConst=" << v->value();
      break;
    }
    case Value::Type::ICONST: {
      const auto* v = static_cast<const IConstValue*>(this);
      os << "IConst=" << v->value();
      break;
    }
    case Value::Type::FUNCTION: {
      // const auto* v = static_cast<const FunctionValue*>(this);
      os << "Function";
      break;
    }
    case Value::Type::CONTRACTION: {
      // const auto* v = static_cast<const ContractionValue*>(this);
      os << "Contraction";
      break;
    }
    default:
      os << "(Unknown)";
      break;
  }
  os << "[this=" << this << ", dims=" << num_dims() << ']';
}

}  // namespace lang
}  // namespace tile
}  // namespace vertexai
