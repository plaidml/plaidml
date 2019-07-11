// Copyright 2019 Intel Corporation.

#include "tile/lang/ast/ast.h"

#include <boost/format.hpp>

#include "base/util/logging.h"
#include "base/util/lookup.h"
#include "base/util/stream_container.h"
#include "tile/lang/ast/ast_ops.h"
#include "tile/lang/ast/fold.h"
#include "tile/lang/ast/traversal.h"
#include "tile/lang/gen_special.h"

namespace vertexai {
namespace tile {
namespace lang {
namespace ast {

std::ostream& operator<<(std::ostream& os, const Expr* expr) {
  os << expr->str() << ": " << static_cast<const void*>(expr);
  return os;
}

std::string to_string(const Expr* expr) {
  std::stringstream ss;
  ss << expr;
  return ss.str();
}

std::string LogicalDim::str() const {
  if (expr) {
    return expr->str();
  }
  return "(null)";
}

std::ostream& operator<<(std::ostream& os, const LogicalDim& dim) {
  os << dim.str();
  return os;
}

std::ostream& operator<<(std::ostream& os, const LogicalShape& shape) {
  os << shape.str();
  return os;
}

bool MergeDims(LogicalDim* lhs, const LogicalDim& rhs) {
  IVLOG(6, "MergeDims> " << *lhs << ", " << rhs);
  auto lhs_int = std::dynamic_pointer_cast<DimIntExpr>(lhs->expr);
  auto rhs_int = std::dynamic_pointer_cast<DimIntExpr>(rhs.expr);
  if (!lhs_int && !rhs_int) {
    // choose either: if both are symbolic
    return true;
  }
  if (lhs_int && rhs_int && lhs_int->value == rhs_int->value) {
    // choose either: if both are 1
    return true;
  }
  if (rhs_int && rhs_int->value == 1) {
    // choose lhs: if rhs == 1
    return true;
  }
  if (lhs_int && lhs_int->value == 1) {
    // choose rhs: if lhs == 1
    *lhs = rhs;
    return true;
  }
  if (lhs_int && !rhs_int) {
    // choose lhs: if rhs is symbolic (assume that symbol will resolve to lhs)
    return true;
  }
  if (rhs_int && !lhs_int) {
    // choose rhs: if lhs is symbolic (assume that symbol will resolve to rhs)
    *lhs = rhs;
    return true;
  }
  return false;
}

bool MergeShapes(LogicalShape* into, const LogicalShape& shape) {
  IVLOG(6, "MergeShapes> " << *into << ", " << shape);
  into->dtype = CommonSupertype(into->dtype, shape.dtype);
  if (shape.dims.size()) {
    if (into->dims.empty()) {
      into->dims = shape.dims;
      return false;
    }
    IVLOG(6, "  Checking compatibility between " << into->dims << " and " << shape.dims);
    auto src = shape.dims.rbegin();
    auto dst = into->dims.rbegin();
    for (;; ++dst, ++src) {
      if (src == shape.dims.rend()) {
        IVLOG(6, "  src broadcasts to dst");
        break;
      }
      if (dst == into->dims.rend()) {
        // Anything that was used to produce 'into' can be broadcast to 'src'.
        // We just need to augment 'into' with the remaining elements of 'src'.
        into->dims.insert(into->dims.begin(), shape.dims.begin(),
                          shape.dims.begin() + (shape.dims.size() - (src - shape.dims.rbegin())));
        IVLOG(6, "  dst broadcasts to src; dims = " << into->dims);
        break;
      }
      if (!MergeDims(&*dst, *src)) {
        // Otherwise, broadcasting cannot be done.
        throw std::runtime_error(
            str(boost::format("Mismatched tensor shapes in elementwise operation: %1% can't match %2%")  //
                % StreamContainer(into->dims) % StreamContainer(shape.dims)));
      }
    }
    IVLOG(6, "  Broadcast possible; LCM dims=" << into->dims);
    return true;
  }
  return false;
}

DataType ComputeOutputType(const std::vector<DataType>& dtypes) {
  DataType ret{DataType::INVALID};
  for (const auto& dtype : dtypes) {
    ret = CommonSupertype(ret, dtype);
  }
  return ret;
}

LogicalShape ComputeOutputShape(const std::vector<ExprPtr>& args) {
  LogicalShape ret;
  for (const auto& arg : args) {
    MergeShapes(&ret, arg->shape);
  }
  return ret;
}

class DimExprEvaluator : public DimVisitor {
  int64_t Visit(const DimIntExpr& expr) final { return expr.value; }

  int64_t Visit(const DimOpExpr& expr) final {
    if (expr.op == IntOp::Neg) {
      if (expr.operands.size() != 1) {
        throw std::runtime_error("Invalid number of operands in DimOpExpr");
      }
      return -expr.operands[0]->Accept(this);
    }
    if (expr.operands.size() != 2) {
      throw std::runtime_error("Invalid number of operands in DimOpExpr");
    }
    auto lhs = expr.operands[0]->Accept(this);
    auto rhs = expr.operands[1]->Accept(this);
    switch (expr.op) {
      case IntOp::Add:
        return lhs + rhs;
      case IntOp::Sub:
        return lhs - rhs;
      case IntOp::Mul:
        return lhs * rhs;
      case IntOp::Div:
        return lhs / rhs;
      default:
        throw std::runtime_error("Unknown DimOp");
    }
  }

  int64_t Visit(const DimNoneExpr&) final { throw std::runtime_error("None value during DimExpr evaluation."); }

  int64_t Visit(const DimRefExpr& expr) final {
    if (!expr.ref) {
      throw std::runtime_error("Undefined ref in DimRefExpr");
    }
    auto dim_expr = expr.ref->shape.dims.at(expr.dim).expr;
    return dim_expr->Accept(this);
  }
};

TensorShape IntoTensorShape(const LogicalShape& shape) {
  DimExprEvaluator dim_eval;
  std::vector<size_t> sizes;
  for (const auto& dim : shape.dims) {
    sizes.push_back(dim.expr->Accept(&dim_eval));
  }
  return SimpleShape(shape.dtype, sizes);
}

class ShapeEvaluator : public AstVisitor<void> {
 public:
  explicit ShapeEvaluator(std::unordered_map<const Expr*, Binding>* bindings) : bindings_by_expr_(bindings) {}

 private:
  void Visit(const ParamExpr& expr) final {  //
    bindings_by_expr_->emplace(&expr, Binding{IntoTensorShape(expr.shape)});
  }

  void Visit(const CallExpr& expr) final {  //
    bindings_by_expr_->emplace(&expr, Binding{IntoTensorShape(expr.shape)});
  }

  void Visit(const ContractionExpr& expr) final {
    bindings_by_expr_->emplace(&expr, Binding{IntoTensorShape(expr.shape)});
  }

  void Visit(const FloatConst& expr) final {
    bindings_by_expr_->emplace(&expr, Binding(expr.value, DataType::FLOAT32));
  }

  void Visit(const IntConst& expr) final {  //
    bindings_by_expr_->emplace(&expr, Binding{expr.value});
  }

  void Visit(const DimExprExpr& expr) final {
    DimExprEvaluator dim_eval;
    auto value = expr.expr->Accept(&dim_eval);
    bindings_by_expr_->emplace(&expr, Binding{value});
  }

 private:
  std::unordered_map<const Expr*, Binding>* bindings_by_expr_;
};

class PolyEvaluator : public PolyVisitor {
 private:
  Polynomial Visit(const PolyDimExpr& expr) final {
    DimExprEvaluator dim_eval;
    return Polynomial(expr.expr->Accept(&dim_eval));
  }

  Polynomial Visit(const PolyIndex& expr) final {
    auto it = seen_.find(expr.idx_id);
    if (it == seen_.end()) {
      auto name = expr.name;
      if (name.empty()) {
        name = NewIdx();
      }
      std::tie(it, std::ignore) = seen_.emplace(expr.idx_id, name);
    }
    return Polynomial(it->second);
  }

  Polynomial Visit(const PolyLiteral& expr) final { return Polynomial(expr.value); }

  Polynomial Visit(const PolyOpExpr& expr) final {
    if (expr.op == IntOp::Neg) {
      if (expr.operands.size() != 1) {
        throw std::runtime_error("Invalid number of operands in PolyOpExpr");
      }
      return -expr.operands[0]->Accept(this);
    }
    if (expr.operands.size() != 2) {
      throw std::runtime_error("Invalid number of operands in PolyOpExpr");
    }
    auto lhs = expr.operands[0]->Accept(this);
    auto rhs = expr.operands[1]->Accept(this);
    switch (expr.op) {
      case IntOp::Add:
        return lhs + rhs;
      case IntOp::Sub:
        return lhs - rhs;
      case IntOp::Mul:
        if (lhs.isConstant()) {
          return rhs * lhs.constant();
        }
        if (rhs.isConstant()) {
          return lhs * rhs.constant();
        }
        throw std::runtime_error(str(boost::format("Non-linear polynomial: %1% * %2%") % lhs % rhs));
      case IntOp::Div:
        if (!rhs.isConstant()) {
          throw std::runtime_error(
              str(boost::format("Divisor of polynomials must be a constant: %1% / %2%") % lhs % rhs));
        }
        return lhs / rhs.constant();
      default:
        throw std::runtime_error("Unknown PolyOp");
        break;
    }
  }

 private:
  std::string NewIdx() { return str(boost::format("x%1%") % next_++); }

 private:
  std::unordered_map<size_t, std::string> seen_;
  std::vector<Polynomial> constraints_;
  size_t next_ = 0;
};

class ProgramEvaluator : public AstVisitor<void> {
 public:
  explicit ProgramEvaluator(const std::string& name) { eval_.runinfo.program_name = name; }

  ProgramEvaluation Evaluate(const std::vector<ExprPtr>& outputs) {
    ShapeEvaluator evaluator(&bindings_by_expr_);
    // Traverse the entire graph in least-dependent to most-dependent order.
    auto ast = FlattenAst(outputs);
    for (const auto& expr : ast) {
      expr->Accept(&evaluator);
      expr->Accept(this);
    }
    for (const auto& expr : outputs) {
      // At this point, it should be guaranteed that the output expressions have been visited.
      auto name = safe_at(&names_by_expr_, expr.get());
      auto shape = safe_at(&bindings_by_expr_, expr.get()).shape;
      IVLOG(2, "Output> " << name << ": " << shape);
      eval_.runinfo.output_shapes.emplace(name, shape);
      eval_.runinfo.program.outputs.push_back(name);
      eval_.outputs.push_back(expr);
    }
    for (const auto& kvp : names_by_expr_) {
      auto name = kvp.second;
      auto binding = safe_at(&bindings_by_expr_, kvp.first);
      eval_.runinfo.vars.emplace(name, binding);
    }
    eval_.runinfo.code = to_string(eval_.runinfo.program);
    eval_.runinfo.from_edsl = true;
    IVLOG(2, "ProgramEvaluator::Evaluate>\n" << eval_.runinfo.code);
    return eval_;
  }

 private:
  void Visit(const ParamExpr& expr) final {
    IVLOG(4, "ProgramEvaluator::Visit> " << to_string(&expr));
    auto name = NewTmp(expr);
    Input input{Input::FIXED, name};
    for (size_t i = 0; i < expr.shape.dims.size(); i++) {
      auto dim_name = str(boost::format("%1%_%2%") % name % i);
      input.dims.emplace_back(dim_name);
    }
    auto shape = safe_at(&bindings_by_expr_, &expr).shape;
    eval_.inputs.push_back(&expr);
    eval_.runinfo.program.inputs.push_back(input);
    eval_.runinfo.input_shapes.emplace(name, shape);
    names_by_expr_.emplace(&expr, name);
  }

  void Visit(const FloatConst& expr) final {
    IVLOG(4, "ProgramEvaluator::Visit> " << to_string(&expr));
    auto name = NewTmp(expr);
    Op op{
        Op::CONSTANT,                  // tag
        name,                          // output
        {std::to_string(expr.value)},  // inputs
        {},                            // Contraction
        {"fconst"},                    // Function
    };
    eval_.runinfo.program.ops.emplace_back(op);
    names_by_expr_.emplace(&expr, name);
  }

  void Visit(const IntConst& expr) final {
    IVLOG(4, "ProgramEvaluator::Visit> " << to_string(&expr));
    auto name = NewTmp(expr);
    Op op{
        Op::CONSTANT,                  // tag
        name,                          // output
        {std::to_string(expr.value)},  // inputs
        {},                            // Contraction
        {"iconst"},                    // Function
    };
    eval_.runinfo.program.ops.emplace_back(op);
    names_by_expr_.emplace(&expr, name);
  }

  void Visit(const DimExprExpr& expr) final {
    IVLOG(4, "ProgramEvaluator::Visit> " << to_string(&expr));
    DimExprEvaluator dim_eval;
    auto value = expr.expr->Accept(&dim_eval);
    auto name = NewTmp(expr);
    Op op{
        Op::CONSTANT,             // tag
        name,                     // output
        {std::to_string(value)},  // inputs
        {},                       // Contraction
        {"iconst"},               // Function
    };
    eval_.runinfo.program.ops.emplace_back(op);
    names_by_expr_.emplace(&expr, name);
  }

  void Visit(const CallExpr& expr) final {
    IVLOG(4, "ProgramEvaluator::Visit> " << to_string(&expr));
    std::vector<std::string> args;
    for (const auto& arg : expr.args) {
      args.emplace_back(safe_at(&names_by_expr_, arg.get()));
    }
    auto name = NewTmp(expr);
    Op op{
        Op::FUNCTION,  // tag
        name,          // output
        args,          // inputs
        {},            // Contraction
        {expr.fn},     // Function
    };
    eval_.runinfo.program.ops.emplace_back(op);
    names_by_expr_.emplace(&expr, name);
  }

  void Visit(const ContractionExpr& expr) final {
    IVLOG(4, "ProgramEvaluator::Visit> " << to_string(&expr));
    PolyEvaluator poly_eval;
    DimExprEvaluator dim_eval;
    Contraction cion;
    cion.agg_op = expr.agg_op;
    cion.comb_op = expr.combo_op;
    cion.no_defract = expr.no_defract;
    if (expr.use_default) {
      cion.use_default = safe_at(&names_by_expr_, expr.use_default.get());
    }
    cion.specs.emplace_back(TensorSpec{});
    std::vector<std::string> inputs;
    for (const auto& input : expr.inputs) {
      TensorSpec tensor_spec;
      tensor_spec.id = safe_at(&names_by_expr_, input->ref.get());
      inputs.push_back(tensor_spec.id);
      for (const auto& idx : input->index_spec) {
        auto poly = idx->Accept(&poly_eval);
        tensor_spec.spec.push_back(poly);
      }
      cion.specs.emplace_back(tensor_spec);
    }
    auto name = NewTmp(expr);
    cion.specs[0].id = name;
    for (const auto& idx : expr.output->index_spec) {
      auto poly = idx->Accept(&poly_eval);
      cion.specs[0].spec.push_back(poly);
    }
    for (const auto& size_expr : expr.output->output_dims) {
      auto size = size_expr->Accept(&dim_eval);
      cion.output_size.push_back(std::to_string(size));
    }
    for (const auto& constraint : expr.constraints) {
      auto poly = constraint->lhs->Accept(&poly_eval);
      auto range = constraint->rhs->Accept(&dim_eval);
      tile::math::RangeConstraint bound(poly, range);
      cion.constraints.emplace_back(bound);
    }
    Op op{
        Op::CONTRACTION,  // tag
        name,             // output
        inputs,           // inputs
        cion,             // Contraction
    };
    eval_.runinfo.program.ops.emplace_back(op);
    names_by_expr_.emplace(&expr, name);
  }

 private:
  // The current algorithm works by making all unnamed nodes automatically
  // generated so that they are unique but only if names that begin with
  // underscore ("_") are reserved by the system.
  std::string NewTmp(const Expr& expr) {
    if (expr.name.empty()) {
      return str(boost::format("_X%1%") % eval_.runinfo.program.next_tmp++);
    }
    return MakeUniqueName(expr.name);
  }

  std::string MakeUniqueName(const std::string& prefix) {
    bool is_new;
    auto name = prefix;
    std::tie(std::ignore, is_new) = names_.insert(name);
    for (size_t i = 0; !is_new; i++) {
      name = str(boost::format("%1%%2%") % prefix % i);
      std::tie(std::ignore, is_new) = names_.insert(name);
    }
    return name;
  }

 private:
  std::set<std::string> names_;
  std::unordered_map<const Expr*, std::string> names_by_expr_;
  std::unordered_map<const Expr*, Binding> bindings_by_expr_;
  ProgramEvaluation eval_;
};

class ExprOptimizer : public AstPass {
 private:
  ExprPtr Visit(const CallExpr& expr) final {
    IVLOG(4, "ExprOptimizer::Visit(CallExpr)> " << &expr);
    if (expr.fn == "simple_reduce") {
      return simple_reduce(expr.args);
    }
    return MakeCall(expr.fn, expr.args);
  }

  ExprPtr Visit(const DimExprExpr& expr) final {
    IVLOG(4, "ExprOptimizer::Visit(DimExprExpr)> " << &expr);
    DimExprEvaluator dim_eval;
    auto value = expr.expr->Accept(&dim_eval);
    return std::make_shared<IntConst>(value);
  }

 private:
  ExprPtr simple_reduce(const std::vector<ExprPtr>& args) {
    IVLOG(4, "ExprOptimizer::simple_reduce>");
    if (args.size() != 2) {
      throw std::runtime_error("simple_reduce expects 2 arguments");
    }
    auto deriv = args[0];
    auto expr = args[1];
    IVLOG(4, "  deriv: " << deriv->shape);
    IVLOG(4, "  expr: " << deriv->shape);
    return deriv;
  }
};

ProgramEvaluation Evaluate(const std::string& name, const std::vector<ExprPtr>& outputs) {
  ExprOptimizer optimizer;
  auto new_outputs = RunAstPass(outputs, &optimizer);
  return ProgramEvaluator(name).Evaluate(new_outputs);
}

std::string LogicalShape::str() const {
  std::stringstream ss;
  ss << to_string(dtype);
  ss << "(";
  for (size_t i = 0; i < dims.size(); i++) {
    if (i > 0) {
      ss << ", ";
    }
    ss << dims[i];
  }
  ss << ")";
  if (layout.size()) {
    ss << ":" << layout;
  }
  return ss.str();
}

std::vector<DimExprPtr> LogicalShape::dims_as_exprs() const {
  std::vector<DimExprPtr> ret(dims.size());
  for (size_t i = 0; i < dims.size(); i++) {
    ret[i] = dims[i].expr;
  }
  return ret;
}

void LogicalShape::bind_dims(std::vector<DimExprPtr>* into) {
  if (dims.size() != into->size()) {
    throw std::runtime_error(
        boost::str(boost::format("bind_dims() mismatch. Tensor shape: %1%, dims: %2%") % dims.size() % into->size()));
  }
  for (size_t i = 0; i < dims.size(); i++) {
    auto none_expr = std::dynamic_pointer_cast<DimNoneExpr>(into->at(i));
    if (none_expr) {
      (*into)[i] = dims[i].expr;
    } else {
      auto lhs_int = std::dynamic_pointer_cast<DimIntExpr>(into->at(i));
      auto rhs_int = std::dynamic_pointer_cast<DimIntExpr>(dims[i].expr);
      if (lhs_int && rhs_int) {
        if (lhs_int->value != rhs_int->value) {
          throw std::runtime_error(
              boost::str(boost::format("bind_dims() mismatch on dim %1%. Required: %2%, Actual: %3%") % i %
                         rhs_int->value % lhs_int->value));
        }
      }
    }
  }
}

TupleExpr::TupleExpr(const std::vector<ExprPtr>& exprs) : exprs(exprs) {
  // TODO: compute shape?
}

std::string TupleExpr::str() const {
  std::stringstream ss;
  ss << StreamContainer(exprs);
  return ss.str();
}

IntConst::IntConst(int64_t value) : Expr(LogicalShape{DataType::INT32}), value(value) {
  // TODO: should this be a DataType::INT64?
}

std::string IntConst::str() const {
  std::stringstream ss;
  ss << value;
  return ss.str();
}

FloatConst::FloatConst(double value) : Expr(LogicalShape{DataType::FLOAT32}), value(value) {}

std::string FloatConst::str() const {
  std::stringstream ss;
  ss << value;
  return ss.str();
}

DimExprExpr::DimExprExpr(const DimExprPtr& expr) : Expr(LogicalShape{DataType::INT32}), expr(expr) {
  // TODO: should this be a DataType::INT64?
}

std::string DimExprExpr::str() const { return expr->str(); }

ParamExpr::ParamExpr(const std::string& name) : Expr(name) {}

void ParamExpr::ComputeShape(const std::shared_ptr<ParamExpr>& ref, const LogicalShape& in_shape) {
  shape.dtype = in_shape.dtype;
  shape.dims.clear();
  for (size_t i = 0; i < in_shape.dims.size(); i++) {
    const auto& dim = in_shape.dims[i];
    auto int_expr = std::dynamic_pointer_cast<DimIntExpr>(dim.expr);
    if (int_expr) {
      if (int_expr->value) {
        shape.dims.push_back(dim);
      } else {
        shape.dims.push_back(LogicalDim{std::make_shared<DimRefExpr>(ref, i)});
      }
    } else {
      shape.dims.push_back(dim);
    }
  }
}

std::string ParamExpr::str() const {
  if (name.size()) {
    return name;
  }
  std::stringstream ss;
  ss << "ParamExpr{" << shape << "}";
  return ss.str();
}

CallExpr::CallExpr(const std::string& fn, const std::vector<ExprPtr>& args)
    : fn(fn),  //
      args(args) {}

void CallExpr::ComputeShape() {
  IVLOG(5, "CallExpr::ComputeShape> fn: " << fn);
  for (const auto& arg : args) {
    IVLOG(5, "  " << arg->shape.str());
  }
  auto op = PrimitiveOpRegistry::Instance()->Resolve(fn);
  if (op) {
    shape = op->ComputeShape(args);
  } else {
    shape = ComputeOutputShape(args);
  }
}

std::string CallExpr::str() const {
  std::stringstream ss;
  ss << fn << "(";
  for (size_t i = 0; i < args.size(); i++) {
    if (i) {
      ss << ", ";
    }
    ss << args[i]->str();
  }
  ss << ")";
  return ss.str();
}

ContractionExpr::ContractionExpr() : Expr(LogicalShape{}) {}

void ContractionExpr::ComputeShape(const std::string& layout) {
  IVLOG(5, "ContractionExpr::ComputeShape> layout: \"" << layout << "\"");
  DataType dtype = DataType::INVALID;
  if (combo_op == CombinationOp::COND) {
    dtype = DataType::BOOLEAN;
  } else {
    std::vector<DataType> dtypes;
    for (const auto& input : inputs) {
      dtypes.push_back(input->ref->shape.dtype);
    }
    dtype = ComputeOutputType(dtypes);
  }
  shape = LogicalShape(dtype, layout);
  for (const auto& size : output->output_dims) {
    shape.dims.push_back(LogicalDim{size});
  }
}

std::string ContractionExpr::str() const {
  std::stringstream ss;
  ss << output->str() << " = ";
  ss << static_cast<char>(agg_op) << "(";
  for (size_t i = 0; i < inputs.size(); i++) {
    if (i) {
      if (i == 1 && combo_op == CombinationOp::COND) {
        ss << " == ";
      } else if (combo_op == CombinationOp::EQ) {
        ss << " == ";
      } else {
        ss << " " << static_cast<char>(combo_op) << " ";
      }
    }
    ss << inputs[i]->str();
  }
  ss << ")";
  for (const auto& constraint : constraints) {
    ss << ", " << constraint->str();
  }
  if (no_defract) {
    ss << " no_defract";
  }
  if (use_default) {
    ss << " default " << use_default->str();
  }
  return ss.str();
}

ConstraintExpr::ConstraintExpr(const PolyExprPtr& lhs, const DimExprPtr& rhs)
    : lhs(lhs),  //
      rhs(rhs) {}

std::string ConstraintExpr::str() const { return "ConstraintExpr"; }

// input ctor
TensorSpecExpr::TensorSpecExpr(  //
    const ExprPtr& ref,          //
    const std::vector<PolyExprPtr>& index_spec)
    : ref(ref),  //
      index_spec(index_spec) {}

// output ctor
TensorSpecExpr::TensorSpecExpr(                  //
    const std::vector<PolyExprPtr>& index_spec,  //
    const std::vector<DimExprPtr>& output_dims)
    : index_spec(index_spec),  //
      output_dims(output_dims) {}

std::string TensorSpecExpr::str() const {
  std::stringstream ss;
  if (name.size()) {
    ss << name;
  } else {
    if (ref) {
      ss << "_";

    } else {
      ss << "O";
    }
  }
  ss << "[";
  for (size_t i = 0; i < index_spec.size(); i++) {
    if (i) {
      ss << ", ";
    }
    ss << index_spec[i]->str();
  }
  if (!ref && output_dims.size()) {
    ss << " : ";
    for (size_t i = 0; i < output_dims.size(); i++) {
      if (i) {
        ss << ", ";
      }
      ss << output_dims[i]->str();
    }
  }
  ss << "]";
  return ss.str();
}

std::string PolyIndex::str() const {
  if (name.size()) {
    return name;
  }
  return boost::str(boost::format("x%1%") % idx_id);
}

std::string PolyLiteral::str() const { return std::to_string(value); }

std::string to_string(IntOp op) {
  switch (op) {
    case IntOp::Neg:
      return "-";
    case IntOp::Add:
      return "+";
    case IntOp::Sub:
      return "-";
    case IntOp::Mul:
      return "*";
    case IntOp::Div:
      return "/";
    default:
      throw std::runtime_error("Invalid op");
  }
}

std::string PolyOpExpr::str() const {
  std::stringstream ss;
  switch (op) {
    case IntOp::Neg:
      ss << "(-" << operands[0]->str() << ")";
      break;
    case IntOp::Add:
    case IntOp::Sub:
    case IntOp::Mul:
    case IntOp::Div:
      ss << "(" << operands[0]->str() << " " << to_string(op) << " " << operands[1]->str() << ")";
      break;
  }
  return ss.str();
}

std::string DimIntExpr::str() const {
  std::stringstream ss;
  ss << value;
  return ss.str();
}

std::string DimOpExpr::str() const {
  std::stringstream ss;
  switch (op) {
    case IntOp::Neg:
      ss << "(-" << operands[0]->str() << ")";
      break;
    case IntOp::Add:
    case IntOp::Sub:
    case IntOp::Mul:
    case IntOp::Div:
      ss << "(" << operands[0]->str() << " " << to_string(op) << " " << operands[1]->str() << ")";
      break;
  }
  return ss.str();
}

std::string DimRefExpr::str() const {
  std::stringstream ss;
  ss << ref->str() << "[" << dim << "]";
  return ss.str();
}

std::string PolyDimExpr::str() const { return expr->str(); }

}  // namespace ast
}  // namespace lang
}  // namespace tile
}  // namespace vertexai
