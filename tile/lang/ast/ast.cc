// Copyright 2019 Intel Corporation.

#include "tile/lang/ast/ast.h"

#include <set>

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
  explicit ProgramEvaluator(const std::string& name) {  //
    eval_.runinfo.program_name = name;
  }

  ProgramEvaluation Evaluate(const ProgramMutations& mutations) {
    ShapeEvaluator evaluator(&bindings_by_expr_);
    // Traverse the entire graph in least-dependent to most-dependent order.
    auto ast = FlattenAst(mutations);
    for (const auto& expr : ast) {
      expr->Accept(&evaluator);
      expr->Accept(this);
    }
    for (const auto& expr : mutations.outputs) {
      // At this point, it should be guaranteed that the output expressions have been visited.
      auto name = safe_at(&eval_.names_by_expr, expr.get());
      auto shape = safe_at(&bindings_by_expr_, expr.get()).shape;
      IVLOG(2, "Output> " << name << ": " << shape);
      eval_.runinfo.output_shapes.emplace(name, shape);
      eval_.runinfo.program.outputs.push_back(name);
      eval_.outputs.push_back(expr);
    }
    for (const auto& update : mutations.updates) {
      auto src_name = safe_at(&eval_.names_by_expr, update.src.get());
      auto src_shape = safe_at(&bindings_by_expr_, update.src.get()).shape;
      auto dst_expr = std::dynamic_pointer_cast<ParamExpr>(update.dst);
      if (!dst_expr) {
        throw std::runtime_error("Updates can only have ParamExpr destinations.");
      }
      IVLOG(2, "Update> " << src_name << ": " << src_shape);
      eval_.updates.emplace(src_name, dst_expr.get());
      eval_.runinfo.output_shapes.emplace(src_name, src_shape);
      eval_.runinfo.program.outputs.push_back(src_name);
    }
    for (const auto& kvp : eval_.names_by_expr) {
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
    eval_.names_by_expr.emplace(&expr, name);
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
    eval_.names_by_expr.emplace(&expr, name);
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
    eval_.names_by_expr.emplace(&expr, name);
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
    eval_.names_by_expr.emplace(&expr, name);
  }

  void Visit(const CallExpr& expr) final {
    IVLOG(4, "ProgramEvaluator::Visit> " << to_string(&expr));
    if (expr.fn == "prng") {
      ProcessPrng(expr);
      return;
    }
    std::vector<std::string> args;
    for (const auto& arg : expr.args) {
      args.emplace_back(safe_at(&eval_.names_by_expr, arg.get()));
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
    eval_.names_by_expr.emplace(&expr, name);
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
      cion.use_default = safe_at(&eval_.names_by_expr, expr.use_default.get());
    }
    cion.specs.emplace_back(TensorSpec{});
    std::vector<std::string> inputs;
    for (const auto& src : expr.srcs) {
      TensorSpec tensor_spec;
      tensor_spec.id = safe_at(&eval_.names_by_expr, src->ref.get());
      inputs.push_back(tensor_spec.id);
      for (const auto& idx : src->idxs) {
        auto poly = idx->Accept(&poly_eval);
        tensor_spec.spec.push_back(poly);
      }
      cion.specs.emplace_back(tensor_spec);
    }
    auto name = NewTmp(expr);
    cion.specs[0].id = name;
    for (const auto& idx : expr.sink_idxs->idxs) {
      auto poly = idx->Accept(&poly_eval);
      cion.specs[0].spec.push_back(poly);
    }
    for (const auto& size_expr : expr.sink_dims->dims) {
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
    eval_.names_by_expr.emplace(&expr, name);
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

  void ProcessPrng(const CallExpr& expr) {
    auto step = NewTmp(expr);
    auto state = NewTmp(expr);
    auto value = NewTmp(expr);
    std::vector<std::string> step_args(expr.args.size());
    for (size_t i = 0; i < step_args.size(); i++) {
      if (i == 0) {
        auto state_expr = std::dynamic_pointer_cast<ParamExpr>(expr.args[i]);
        if (!state_expr) {
          throw std::runtime_error("First argument to prng() must be a ParamExpr");
        }
        eval_.updates.emplace(state, state_expr.get());
      }
      step_args[i] = safe_at(&eval_.names_by_expr, expr.args[i].get());
    }
    eval_.runinfo.output_shapes.emplace(state, SimpleShape(DataType::UINT32, {3, k_rng_size}));
    eval_.runinfo.program.outputs.push_back(state);
    eval_.runinfo.program.ops.emplace_back(Op{
        Op::FUNCTION,   // tag
        step,           // output
        step_args,      // inputs
        {},             // Contraction
        {"prng_step"},  // Function
    });
    eval_.runinfo.program.ops.emplace_back(Op{
        Op::FUNCTION,    // tag
        state,           // output
        {step},          // inputs
        {},              // Contraction
        {"prng_state"},  // Function
    });
    eval_.runinfo.program.ops.emplace_back(Op{
        Op::FUNCTION,    // tag
        value,           // output
        {step},          // inputs
        {},              // Contraction
        {"prng_value"},  // Function
    });
    eval_.names_by_expr.emplace(&expr, value);
  }

 private:
  std::set<std::string> names_;
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
    IVLOG(4, "  expr: " << expr->shape);

    std::vector<PolyExprPtr> srcs;
    std::vector<PolyExprPtr> sink_idxs;
    bool nontrivial_contraction = false;
    auto input_ndims = deriv->shape.dims.size();
    auto input_only_ndims = input_ndims - expr->shape.dims.size();
    for (size_t i = 0; i < deriv->shape.dims.size(); i++) {
      DimExprEvaluator dim_eval;
      auto curr_idx = std::make_shared<PolyIndex>(i);
      srcs.push_back(curr_idx);
      if (i < input_only_ndims) {
        // Add the initial indexes only to the output spec; only once lengths are aligned look at both
        nontrivial_contraction = true;  // If the lengths don't match this is nontrivial
      } else {
        auto output_i = i - input_only_ndims;
        // Where the output dim is 1, put a 0 in the output spec
        // Where the output dim is >1, put the index from the corresponding axis of the input_spec
        if (expr->shape.dims[output_i].expr->Accept(&dim_eval) == 1) {
          sink_idxs.push_back(std::make_shared<PolyLiteral>(0));
          if (deriv->shape.dims[i].expr->Accept(&dim_eval) != 1) {
            nontrivial_contraction = true;
          }
        } else {
          sink_idxs.push_back(curr_idx);
        }
      }
    }

    if (nontrivial_contraction) {
      auto dop = std::make_shared<ContractionExpr>();
      dop->agg_op = AggregationOp::SUM;
      dop->combo_op = CombinationOp::NONE;
      dop->srcs.push_back(std::make_shared<IndexMapExpr>(deriv, srcs));
      dop->sink_idxs = std::make_shared<IndexMapExpr>(nullptr, sink_idxs);
      dop->sink_dims = std::make_shared<SizeMapExpr>(expr->shape.dims_as_exprs());
      dop->ComputeShape(expr->shape.layout);
      return dop;
    } else {
      return deriv;
    }
  }
};

ProgramEvaluation Evaluate(const std::string& name, ProgramMutations mutations) {
  std::set<Expr*> dups;
  for (size_t i = 0; i < mutations.outputs.size(); i++) {
    auto output = mutations.outputs[i];
    auto ptr = output.get();
    if (std::dynamic_pointer_cast<ParamExpr>(output) || dups.count(ptr)) {
      mutations.outputs[i] = MakeCall("ident", {output});
    }
    dups.insert(ptr);
  }
  ExprOptimizer optimizer;
  return ProgramEvaluator(name).Evaluate(RunAstPass(mutations, &optimizer));
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
    if (auto none_expr = std::dynamic_pointer_cast<DimNoneExpr>(into->at(i))) {
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
  ss << "int(" << value << ")";
  return ss.str();
}

FloatConst::FloatConst(double value) : Expr(LogicalShape{DataType::FLOAT32}), value(value) {}

std::string FloatConst::str() const {
  std::stringstream ss;
  ss << "float(" << value << ")";
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
    if (auto int_expr = std::dynamic_pointer_cast<DimIntExpr>(dim.expr)) {
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
  ss << fn << "()";
  return ss.str();
}

ContractionExpr::ContractionExpr() : Expr(LogicalShape{}) {}

void ContractionExpr::ComputeShape(const std::string& layout) {
  IVLOG(5, "ContractionExpr::ComputeShape> layout: \"" << layout << "\"");
  DataType dtype = DataType::INVALID;
  if (combo_op == CombinationOp::COND) {
    if (srcs.size() != 3) {
      throw std::runtime_error("Internal error: Invalid number of inputs for COND");
    }
    dtype = srcs[2]->ref->shape.dtype;
  } else {
    std::vector<DataType> dtypes;
    for (const auto& src : srcs) {
      dtypes.push_back(src->ref->shape.dtype);
    }
    dtype = ComputeOutputType(dtypes);
  }
  shape = LogicalShape(dtype, layout);
  for (const auto& size : sink_dims->dims) {
    shape.dims.push_back(LogicalDim{size});
  }
}

std::string ContractionExpr::str() const {
  std::stringstream ss;
  ss << sink_idxs->str() << ":" << sink_dims->str() << " = ";
  ss << static_cast<char>(agg_op) << "(";
  for (size_t i = 0; i < srcs.size(); i++) {
    if (i) {
      if (i == 1 && combo_op == CombinationOp::COND) {
        ss << " == ";
      } else if (combo_op == CombinationOp::EQ) {
        ss << " == ";
      } else {
        ss << " " << static_cast<char>(combo_op) << " ";
      }
    }
    ss << srcs[i]->str();
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

IndexMapExpr::IndexMapExpr(  //
    const ExprPtr& ref,      //
    const std::vector<PolyExprPtr>& idxs)
    : ref(ref),  //
      idxs(idxs) {}

std::string IndexMapExpr::str() const {
  std::stringstream ss;
  if (name.size()) {
    ss << name;
  } else {
    ss << "_";
  }
  ss << "[";
  for (size_t i = 0; i < idxs.size(); i++) {
    if (i) {
      ss << ", ";
    }
    ss << idxs[i]->str();
  }
  ss << "]";
  return ss.str();
}

SizeMapExpr::SizeMapExpr(  //
    const std::vector<DimExprPtr>& dims)
    : dims(dims) {}

std::string SizeMapExpr::str() const {
  std::stringstream ss;
  ss << "[";
  for (size_t i = 0; i < dims.size(); i++) {
    if (i) {
      ss << ", ";
    }
    ss << dims[i]->str();
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
