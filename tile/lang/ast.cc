#include "tile/lang/ast.h"

#include <boost/format.hpp>

#include "base/util/logging.h"
#include "base/util/lookup.h"
#include "base/util/stream_container.h"
#include "tile/lang/ast_ops.h"
#include "tile/lang/gen_special.h"

namespace vertexai {
namespace tile {
namespace lang {

using AstVector = std::vector<std::shared_ptr<Expr>>;
using Polynomial = math::Polynomial<math::Rational>;

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
  IVLOG(4, "MergeDims> " << *lhs << ", " << rhs);
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
  IVLOG(4, "MergeShapes> " << *into << ", " << shape);
  into->dtype = CommonSupertype(into->dtype, shape.dtype);
  if (shape.dims.size()) {
    if (into->dims.empty()) {
      into->dims = shape.dims;
      return false;
    }
    IVLOG(4, "  Checking compatibility between " << into->dims << " and " << shape.dims);
    auto src = shape.dims.rbegin();
    auto dst = into->dims.rbegin();
    for (;; ++dst, ++src) {
      if (src == shape.dims.rend()) {
        IVLOG(4, "  src broadcasts to dst");
        break;
      }
      if (dst == into->dims.rend()) {
        // Anything that was used to produce 'into' can be broadcast to 'src'.
        // We just need to augment 'into' with the remaining elements of 'src'.
        into->dims.insert(into->dims.begin(), shape.dims.begin(),
                          shape.dims.begin() + (shape.dims.size() - (src - shape.dims.rbegin())));
        IVLOG(4, "  dst broadcasts to src; dims = " << into->dims);
        break;
      }
      if (!MergeDims(&*dst, *src)) {
        // Otherwise, broadcasting cannot be done.
        throw std::runtime_error(
            str(boost::format("Mismatched tensor shapes in elementwise operation: %1% can't match %2%")  //
                % StreamContainer(into->dims) % StreamContainer(shape.dims)));
      }
    }
    IVLOG(4, "  Broadcast possible; LCM dims=" << into->dims);
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

LogicalShape ComputeOutputShape(const std::vector<std::shared_ptr<Expr>>& args) {
  LogicalShape ret;
  for (const auto& arg : args) {
    MergeShapes(&ret, arg->shape);
  }
  return ret;
}

class AstTraversal : public AstVisitor {
 public:
  explicit AstTraversal(const std::vector<std::shared_ptr<Expr>>& exprs) {
    for (const auto& expr : exprs) {
      Push(expr);
    }
    while (stack_.size()) {
      auto entry = stack_.top();
      stack_.pop();
      if (entry.second) {
        flat_.push_back(entry.first);
      } else if (!seen_.count(entry.first.get())) {
        seen_.insert(entry.first.get());
        stack_.push(std::make_pair(entry.first, true));
        entry.first->Accept(this);
      }
    }
    IVLOG(4, "AstTraversal: " << StreamContainer(flat_));
  }

  const AstVector& flat() const { return flat_; }

 private:
  void Visit(const CallExpr& expr) {
    // push arguments from right-to-left so they eventually get processed in left-to-right order
    for (auto it = expr.args.rbegin(); it != expr.args.rend(); ++it) {
      Push(*it);
    }
  }

  void Visit(const ConstraintExpr& expr) { throw std::runtime_error("ConstraintExpr visitor not implemented"); }

  void Visit(const ContractionExpr& expr) {
    // push inputs from right-to-left so they eventually get processed in left-to-right order
    for (auto it = expr.inputs.rbegin(); it != expr.inputs.rend(); ++it) {
      Push((*it)->ref);
    }
    if (expr.use_default) {
      Push(expr.use_default);
    }
  }

  void Visit(const FloatConst& expr) {}

  void Visit(const IntConst& expr) {}

  void Visit(const ParamExpr& expr) {}

  void Visit(const DimExprExpr& expr) {}

  void Visit(const TensorSpecExpr& expr) { throw std::runtime_error("TensorSpecExpr visitor not implemented"); }

 private:
  void Push(const std::shared_ptr<Expr>& expr) {
    if (!expr) {
      throw std::runtime_error("Invalid expression");
    }
    IVLOG(4, "AstTraversal::Push> " << expr.get());
    stack_.push(std::make_pair(expr, false));
  }

 private:
  std::stack<std::pair<std::shared_ptr<Expr>, bool>> stack_;
  AstVector flat_;
  std::unordered_set<const Expr*> seen_;
};

class DimExprEvaluator : public DimVisitor {
  int64_t Visit(const DimIntExpr& expr) { return expr.value; }

  int64_t Visit(const DimOpExpr& expr) {
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

  int64_t Visit(const DimNoneExpr&) { throw std::runtime_error("None value during DimExpr evaluation."); }

  int64_t Visit(const DimRefExpr& expr) {
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

class ShapeEvaluator : public AstVisitor {
 public:
  explicit ShapeEvaluator(std::unordered_map<const Expr*, Binding>* bindings) : bindings_by_expr_(bindings) {}

 private:
  void Visit(const ParamExpr& expr) { bindings_by_expr_->emplace(&expr, Binding{IntoTensorShape(expr.shape)}); }
  void Visit(const CallExpr& expr) { bindings_by_expr_->emplace(&expr, Binding{IntoTensorShape(expr.shape)}); }
  void Visit(const ConstraintExpr&) { throw std::runtime_error("Not implemented"); }
  void Visit(const ContractionExpr& expr) { bindings_by_expr_->emplace(&expr, Binding{IntoTensorShape(expr.shape)}); }
  void Visit(const FloatConst& expr) { bindings_by_expr_->emplace(&expr, Binding(expr.value, DataType::FLOAT32)); }
  void Visit(const IntConst& expr) { bindings_by_expr_->emplace(&expr, Binding{expr.value}); }

  void Visit(const DimExprExpr& expr) {
    DimExprEvaluator dim_eval;
    auto value = expr.expr->Accept(&dim_eval);
    bindings_by_expr_->emplace(&expr, Binding{value});
  }

  void Visit(const TensorSpecExpr& expr) { throw std::runtime_error("Not implemented"); }

 private:
  std::unordered_map<const Expr*, Binding>* bindings_by_expr_;
};

class PolyEvaluator : public PolyVisitor {
 private:
  Polynomial Visit(const PolyDimExpr& expr) {
    DimExprEvaluator dim_eval;
    return Polynomial(expr.expr->Accept(&dim_eval));
  }

  Polynomial Visit(const PolyIndex& expr) {
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

  Polynomial Visit(const PolyLiteral& expr) { return Polynomial(expr.value); }

  Polynomial Visit(const PolyOpExpr& expr) {
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

class Evaluator : public AstVisitor {
 public:
  explicit Evaluator(const std::string& name) { eval_.runinfo.program_name = name; }

  ProgramEvaluation Evaluate(const std::vector<std::shared_ptr<Expr>>& exprs) {
    ShapeEvaluator evaluator(&bindings_by_expr_);
    // Traverse the entire graph in least-dependent to most-dependent order.
    AstTraversal traversal(exprs);
    for (const auto& expr : traversal.flat()) {
      expr->Accept(&evaluator);
      expr->Accept(this);
    }
    for (const auto& expr : exprs) {
      // At this point, it should be guaranteed that the output expressions have been visited.
      auto name = safe_at(&names_by_expr_, expr.get());
      auto shape = safe_at(&bindings_by_expr_, expr.get()).shape;
      IVLOG(2, "Output> " << name << ": " << shape);
      eval_.runinfo.output_shapes.emplace(name, shape);
      eval_.runinfo.program.outputs.push_back(name);
      eval_.outputs.push_back(expr.get());
    }
    for (const auto& kvp : names_by_expr_) {
      auto name = kvp.second;
      auto binding = safe_at(&bindings_by_expr_, kvp.first);
      eval_.runinfo.vars.emplace(name, binding);
    }
    eval_.runinfo.code = to_string(eval_.runinfo.program);
    eval_.runinfo.from_edsl = true;
    IVLOG(2, "Evaluator::Evaluate>\n" << eval_.runinfo.code);
    return eval_;
  }

 private:
  void Visit(const ParamExpr& expr) {
    IVLOG(4, "Evaluator::Visit> " << to_string(&expr));
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

  void Visit(const FloatConst& expr) {
    IVLOG(4, "Evaluator::Visit> " << to_string(&expr));
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

  void Visit(const IntConst& expr) {
    IVLOG(4, "Evaluator::Visit> " << to_string(&expr));
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

  void Visit(const DimExprExpr& expr) {
    IVLOG(4, "Evaluator::Visit> " << to_string(&expr));
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

  void Visit(const CallExpr& expr) {
    IVLOG(4, "Evaluator::Visit> " << to_string(&expr));
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

  void Visit(const ConstraintExpr& expr) { throw std::runtime_error("ConstraintExpr visitor not implemented"); }

  void Visit(const ContractionExpr& expr) {
    IVLOG(4, "Evaluator::Visit> " << to_string(&expr));
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
    for (const auto& size_expr : expr.output->output_sizes) {
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

  void Visit(const TensorSpecExpr& expr) { throw std::runtime_error("TensorSpecExec visitor not implemented"); }

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

ProgramEvaluation Evaluate(const std::string& name, const std::vector<std::shared_ptr<Expr>>& exprs) {
  return Evaluator(name).Evaluate(exprs);
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
  return ss.str();
}

void LogicalShape::bind_dims(std::vector<std::shared_ptr<DimExpr>>* into) {
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

DimExprExpr::DimExprExpr(const std::shared_ptr<DimExpr>& expr) : Expr(LogicalShape{DataType::INT32}), expr(expr) {
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
  ss << "ParamExpr{" << shape.str() << "}";
  return ss.str();
}

CallExpr::CallExpr(const std::string& fn, const std::vector<std::shared_ptr<Expr>>& args) : fn(fn), args(args) {}

void CallExpr::ComputeShape() {
  IVLOG(4, "CallExpr::ComputeShape> fn: " << fn);
  for (const auto& arg : args) {
    IVLOG(4, "  " << arg->shape.str());
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

void ContractionExpr::ComputeShape() {
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
  shape = LogicalShape(dtype);
  for (const auto& size : output->output_sizes) {
    shape.dims.push_back(LogicalDim{size});
  }
}

std::string ContractionExpr::str() const { return "ContractionExpr"; }

ConstraintExpr::ConstraintExpr(const std::shared_ptr<PolyExpr>& lhs, const std::shared_ptr<DimExpr>& rhs)
    : lhs(lhs),  //
      rhs(rhs) {}

std::string ConstraintExpr::str() const { return "ConstraintExpr"; }

TensorSpecExpr::TensorSpecExpr(const std::shared_ptr<Expr>& ref,  //
                               const std::vector<std::shared_ptr<PolyExpr>>& index_spec,
                               const std::vector<std::shared_ptr<DimExpr>>& output_sizes)
    : ref(ref),  //
      index_spec(index_spec),
      output_sizes(output_sizes) {}

std::string TensorSpecExpr::str() const { return "TensorSpecExpr"; }

std::string PolyIndex::str() const {
  if (name.size()) {
    return name;
  }
  return boost::str(boost::format("PolyIndex: %1%") % idx_id);
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

template <typename NumType, typename ExprType>
std::shared_ptr<ExprType> fold_add(const std::shared_ptr<ExprType>& lhs, const std::shared_ptr<ExprType>& rhs) {
  auto lhs_num = std::dynamic_pointer_cast<NumType>(lhs);
  auto rhs_num = std::dynamic_pointer_cast<NumType>(rhs);
  if (lhs_num && rhs_num) {
    return std::make_shared<NumType>(lhs_num->value + rhs_num->value);
  }
  if (lhs_num && lhs_num->value == 0) {
    return rhs;
  }
  if (rhs_num && rhs_num->value == 0) {
    return lhs;
  }
  return nullptr;
}

template <typename NumType, typename OpExprType, typename NegOpType, typename ExprType>
std::shared_ptr<ExprType> fold_sub(const std::shared_ptr<ExprType>& lhs, const std::shared_ptr<ExprType>& rhs,
                                   NegOpType neg_op) {
  auto lhs_num = std::dynamic_pointer_cast<NumType>(lhs);
  auto rhs_num = std::dynamic_pointer_cast<NumType>(rhs);
  if (lhs_num && rhs_num) {
    return std::make_shared<NumType>(lhs_num->value - rhs_num->value);
  }
  // TODO: deal with ComputeShape
  // if (lhs_num && lhs_num->value == 0) {
  //   std::vector<std::shared_ptr<ExprType>> args{rhs};
  //   return std::make_shared<OpExprType>(neg_op, args);
  // }
  if (rhs_num && rhs_num->value == 0) {
    return lhs;
  }
  return nullptr;
}

template <typename NumType, typename ExprType>
std::shared_ptr<ExprType> fold_mul(const std::shared_ptr<ExprType>& lhs, const std::shared_ptr<ExprType>& rhs) {
  auto lhs_num = std::dynamic_pointer_cast<NumType>(lhs);
  auto rhs_num = std::dynamic_pointer_cast<NumType>(rhs);
  if (lhs_num && rhs_num) {
    return std::make_shared<NumType>(lhs_num->value * rhs_num->value);
  }
  if (lhs_num && lhs_num->value == 1) {
    return rhs;
  }
  if (rhs_num && rhs_num->value == 1) {
    return lhs;
  }
  return nullptr;
}

template <typename NumType, typename ExprType>
std::shared_ptr<ExprType> fold_div(const std::shared_ptr<ExprType>& lhs, const std::shared_ptr<ExprType>& rhs) {
  auto lhs_num = std::dynamic_pointer_cast<NumType>(lhs);
  auto rhs_num = std::dynamic_pointer_cast<NumType>(rhs);
  if (lhs_num && rhs_num) {
    return std::make_shared<NumType>(lhs_num->value / rhs_num->value);
  }
  if (lhs_num && lhs_num->value == 0) {
    return std::make_shared<NumType>(0);
  }
  if (rhs_num && rhs_num->value == 1) {
    return lhs;
  }
  return nullptr;
}

template <typename IntType, typename OpType, typename ExprType>
std::shared_ptr<ExprType> MakeOp(IntOp op, const std::vector<std::shared_ptr<ExprType>>& args) {
  switch (op) {
    case IntOp::Neg: {
      auto int_expr = std::dynamic_pointer_cast<IntType>(args[0]);
      if (int_expr) {
        return std::make_shared<IntType>(-int_expr->value);
      }
    } break;
    case IntOp::Add: {
      auto ret = fold_add<IntType>(args[0], args[1]);
      if (ret) {
        return ret;
      }
    } break;
    case IntOp::Sub: {
      auto ret = fold_sub<IntType, OpType>(args[0], args[1], IntOp::Neg);
      if (ret) {
        return ret;
      }
    } break;
    case IntOp::Mul: {
      auto ret = fold_mul<IntType>(args[0], args[1]);
      if (ret) {
        return ret;
      }
    } break;
    case IntOp::Div: {
      auto ret = fold_div<IntType>(args[0], args[1]);
      if (ret) {
        return ret;
      }
    } break;
  }
  return std::make_shared<OpType>(op, args);
}

std::shared_ptr<PolyExpr> MakeOp(IntOp op, const std::vector<std::shared_ptr<PolyExpr>>& args) {
  return MakeOp<PolyLiteral, PolyOpExpr>(op, args);
}

std::shared_ptr<DimExpr> MakeOp(IntOp op, const std::vector<std::shared_ptr<DimExpr>>& args) {
  return MakeOp<DimIntExpr, DimOpExpr>(op, args);
}

std::shared_ptr<Expr> MakeCall(const std::string& fn, const std::vector<std::shared_ptr<Expr>>& args) {
  if (fn == "neg") {
    auto int_expr = std::dynamic_pointer_cast<IntConst>(args[0]);
    if (int_expr) {
      return std::make_shared<IntConst>(-int_expr->value);
    }
    auto float_expr = std::dynamic_pointer_cast<FloatConst>(args[0]);
    if (float_expr) {
      return std::make_shared<FloatConst>(-float_expr->value);
    }
  } else if (fn == "add") {
    auto int_ret = fold_add<IntConst>(args[0], args[1]);
    if (int_ret) {
      return int_ret;
    }
    auto float_ret = fold_add<FloatConst>(args[0], args[1]);
    if (float_ret) {
      return float_ret;
    }
  } else if (fn == "sub") {
    auto int_ret = fold_sub<IntConst, CallExpr>(args[0], args[1], "neg");
    if (int_ret) {
      return int_ret;
    }
    auto float_ret = fold_sub<FloatConst, CallExpr>(args[0], args[1], "neg");
    if (float_ret) {
      return float_ret;
    }
  } else if (fn == "mul") {
    auto int_ret = fold_mul<IntConst>(args[0], args[1]);
    if (int_ret) {
      return int_ret;
    }
    auto float_ret = fold_mul<FloatConst>(args[0], args[1]);
    if (float_ret) {
      return float_ret;
    }
  } else if (fn == "div") {
    auto int_ret = fold_div<IntConst>(args[0], args[1]);
    if (int_ret) {
      return int_ret;
    }
    auto float_ret = fold_div<FloatConst>(args[0], args[1]);
    if (float_ret) {
      return float_ret;
    }
  }
  auto expr = std::make_shared<CallExpr>(fn, args);
  expr->ComputeShape();
  return expr;
}

}  // namespace lang
}  // namespace tile
}  // namespace vertexai
