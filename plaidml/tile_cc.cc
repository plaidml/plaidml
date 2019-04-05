#include "plaidml/tile_cc.h"

#include <algorithm>
#include <iterator>

#include <boost/format.hpp>

#include "base/util/logging.h"
#include "base/util/stream_container.h"
#include "tile/lang/ast.h"
#include "tile/lang/ops.h"
#include "tile/math/polynomial.h"

// TODO:
// - Typed tensors (ndims)
// - Complete all eltwise operations
// - Complete all contraction operations
// - TypeCheck

namespace vertexai {
namespace plaidml {
namespace tile_cc {

using tile::TensorShape;
using Polynomial = tile::math::Polynomial<tile::math::Rational>;
using namespace tile::lang;  // NOLINT

struct Index::Impl {
  std::shared_ptr<PolyExpr> expr;
  std::vector<std::shared_ptr<ConstraintExpr>> constraints;
  Index MakePolyOp(const std::string& op, const Index& rhs);
};

struct Access::Impl {
  std::shared_ptr<Expr> expr;
  Tensor::Impl* src = nullptr;
  void MakeContraction(AggregationOp agg_op, const Access& rhs);
  Access MakeCall(const std::string& fn, const Access& rhs);
};

struct Tensor::Impl {
  std::shared_ptr<Expr> expr;
  tile::TensorShape shape;
};

Index::Index() : impl_(std::make_shared<Impl>()) { impl_->expr = std::make_shared<PolyIndex>(impl_.get()); }

Index::Index(size_t value) : impl_(std::make_shared<Impl>()) { impl_->expr = std::make_shared<PolyLiteral>(value); }

Index::Index(std::unique_ptr<Impl> impl) : impl_(std::move(impl)) {}

Index::~Index() = default;

Index Index::Impl::MakePolyOp(const std::string& op, const Index& rhs) {
  auto impl = std::make_unique<Impl>();
  std::vector<std::shared_ptr<PolyExpr>> operands = {expr, rhs.impl_->expr};
  impl->expr = std::make_shared<PolyOp>(op, operands);
  return Index{std::move(impl)};
}

Index Index::operator+(const Index& rhs) const { return impl_->MakePolyOp("add", rhs); }
Index Index::operator-(const Index& rhs) const { return impl_->MakePolyOp("sub", rhs); }
Index Index::operator*(const Index& rhs) const { return impl_->MakePolyOp("mul", rhs); }
Index Index::operator/(const Index& rhs) const { return impl_->MakePolyOp("div", rhs); }

Constraint Index::operator<(size_t rhs) const {
  auto constraint = std::make_shared<ConstraintExpr>(impl_->expr, rhs);
  impl_->constraints.emplace_back(constraint);
  return Constraint();
}

const Tensor::Impl* Tensor::impl() const { return impl_.get(); }

const tile::TensorShape& Tensor::shape() const { return impl_->shape; }

Tensor::Tensor() : impl_(new Impl) { impl_->expr = std::make_shared<ParamExpr>(0, ""); }

Tensor::Tensor(const TensorShape& shape) : impl_(new Impl) {
  impl_->shape = shape;
  impl_->expr = std::make_shared<ParamExpr>(shape.dims.size(), "");
}

Tensor::Tensor(int value) : impl_(new Impl) { impl_->expr = std::make_shared<IntConst>(value); }

Tensor::Tensor(int64_t value) : impl_(new Impl) { impl_->expr = std::make_shared<IntConst>(value); }

Tensor::Tensor(double value) : impl_(new Impl) { impl_->expr = std::make_shared<FloatConst>(value); }

Tensor::~Tensor() = default;

// Impl Constructor
Tensor::Tensor(std::unique_ptr<Impl> impl) : impl_(std::move(impl)) {}

// Copy Constructor
Tensor::Tensor(const Tensor& rhs) : impl_(new Impl(*rhs.impl_)) {}

// Copy Assignment
Tensor& Tensor::operator=(const Tensor& rhs) {
  if (this != &rhs) {
    impl_.reset(new Impl(*rhs.impl_));
  }
  return *this;
}

Access Tensor::operator()(const std::vector<Index>& idxs, const std::vector<size_t>& sizes) {
  if (idxs.size() != sizes.size()) {
    throw std::runtime_error(
        str(boost::format("OutSpec: idxs and sizes mismatch: %1% != %2%") % idxs.size() % sizes.size()));
  }
  std::vector<std::shared_ptr<PolyExpr>> idx_exprs;
  for (const auto& idx : idxs) {
    idx_exprs.push_back(idx.impl_->expr);
  }
  auto impl = std::make_unique<Access::Impl>();
  impl->src = impl_.get();
  impl->src->shape = tile::SimpleShape(tile::DataType::FLOAT32, sizes);  // FIXME: type promotion
  impl->expr = std::make_shared<TensorSpecExpr>(impl_->expr, idx_exprs, sizes);
  return Access{std::move(impl)};
}

Access Tensor::operator()(const std::vector<Index>& idxs) const {
  if (idxs.size() != impl_->shape.dims.size()) {
    throw std::runtime_error(
        str(boost::format("TensorSpec: wrong number of idxs: %1% != %2%") % idxs.size() % impl_->shape.dims.size()));
  }
  std::vector<std::shared_ptr<PolyExpr>> idx_exprs;
  for (const auto& idx : idxs) {
    idx_exprs.push_back(idx.impl_->expr);
  }
  std::vector<size_t> sizes;
  auto impl = std::make_unique<Access::Impl>();
  impl->src = impl_.get();
  impl->expr = std::make_shared<TensorSpecExpr>(impl_->expr, idx_exprs, sizes);
  return Access{std::move(impl)};
}

size_t Tensor::operator[](const size_t dim) const {
  if (impl_->shape.dims.size() <= dim) {
    throw std::runtime_error("oops: dim out of bounds");
  }
  return impl_->shape.dims[dim].size;
}

Tensor Tensor::operator+(const Tensor& rhs) const { return Call("add", {*this, rhs}); }
Tensor Tensor::operator-(const Tensor& rhs) const { return Call("sub", {*this, rhs}); }
Tensor Tensor::operator*(const Tensor& rhs) const { return Call("mul", {*this, rhs}); }
Tensor Tensor::operator/(const Tensor& rhs) const { return Call("div", {*this, rhs}); }
Tensor Tensor::operator<(const Tensor& rhs) const { return Call("cmp_lt", {*this, rhs}); }
Tensor Tensor::operator>(const Tensor& rhs) const { return Call("cmp_gt", {*this, rhs}); }
Tensor Tensor::operator<=(const Tensor& rhs) const { return Call("cmp_le", {*this, rhs}); }
Tensor Tensor::operator>=(const Tensor& rhs) const { return Call("cmp_ge", {*this, rhs}); }

Tensor& Tensor::no_defract() {
  auto cion_expr = std::dynamic_pointer_cast<ContractionExpr>(impl_->expr);
  if (!cion_expr) {
    throw std::runtime_error("no_defract can only be specified on a contraction");
  }
  cion_expr->no_defract = true;
  return *this;
}

Tensor& Tensor::use_default(const Tensor& rhs) {
  auto cion_expr = std::dynamic_pointer_cast<ContractionExpr>(impl_->expr);
  if (!cion_expr) {
    throw std::runtime_error("use_default can only be specified on a contraction");
  }
  cion_expr->use_default = rhs.impl_->expr;
  return *this;
}

Access::~Access() = default;

Access::Access(std::unique_ptr<Impl> impl) : impl_(std::move(impl)) {}

Access::Access(Access&& rhs) noexcept : impl_(std::move(rhs.impl_)) {}

struct ConstraintCollector : public PolyVisitor {
  Polynomial Visit(const PolyIndex& expr) {
    auto impl = static_cast<const Index::Impl*>(expr.ptr);
    std::copy(impl->constraints.begin(), impl->constraints.end(), std::back_inserter(constraints));
    return Polynomial();
  }

  Polynomial Visit(const PolyLiteral& expr) { return Polynomial(); }

  Polynomial Visit(const PolyOp& expr) {
    for (const auto& op : expr.operands) {
      op->Accept(this);
    }
    return Polynomial();
  }

  std::vector<std::shared_ptr<ConstraintExpr>> constraints;
};

void Access::Impl::MakeContraction(AggregationOp agg_op, const Access& rhs) {
  auto output_spec = std::dynamic_pointer_cast<TensorSpecExpr>(expr);
  if (!output_spec) {
    throw std::runtime_error("oops: out_spec");
  }

  auto cion_expr = std::make_shared<ContractionExpr>();
  cion_expr->agg_op = agg_op;
  cion_expr->output = output_spec;

  auto input_spec = std::dynamic_pointer_cast<TensorSpecExpr>(rhs.impl_->expr);
  if (input_spec) {
    cion_expr->inputs = {input_spec};
  } else {
    auto call_expr = std::dynamic_pointer_cast<CallExpr>(rhs.impl_->expr);
    if (!call_expr) {
      throw std::runtime_error("oops: call_expr");
    }
    if (call_expr->fn == "add") {
      cion_expr->combo_op = CombinationOp::PLUS;
    } else if (call_expr->fn == "mul") {
      cion_expr->combo_op = CombinationOp::MULTIPLY;
    } else if (call_expr->fn == "eq") {
      cion_expr->combo_op = CombinationOp::EQ;
    } else if (call_expr->fn == "cond") {
      cion_expr->combo_op = CombinationOp::COND;
    }
    for (const auto& arg : call_expr->args) {
      auto spec = std::dynamic_pointer_cast<TensorSpecExpr>(arg);
      cion_expr->inputs.push_back(spec);
    }
  }

  ConstraintCollector cc;
  for (const auto& idx : output_spec->index_spec) {
    idx->Accept(&cc);
  }

  for (const auto& tensor : cion_expr->inputs) {
    for (const auto& idx : tensor->index_spec) {
      idx->Accept(&cc);
    }
  }
  cion_expr->constraints = cc.constraints;

  src->expr = cion_expr;
}

Access& Access::operator+=(const Access& rhs) {
  impl_->MakeContraction(AggregationOp::SUM, rhs);
  return *this;
}

Access& Access::operator*=(const Access& rhs) {
  impl_->MakeContraction(AggregationOp::PROD, rhs);
  return *this;
}

Access& Access::operator>=(const Access& rhs) {
  impl_->MakeContraction(AggregationOp::MAX, rhs);
  return *this;
}

Access& Access::operator<=(const Access& rhs) {
  impl_->MakeContraction(AggregationOp::MIN, rhs);
  return *this;
}

Access& Access::operator=(const Access& rhs) {
  impl_->MakeContraction(AggregationOp::ASSIGN, rhs);
  return *this;
}

Access Access::Impl::MakeCall(const std::string& fn, const Access& rhs) {
  auto impl = std::make_unique<Impl>();
  std::vector<std::shared_ptr<Expr>> args = {expr, rhs.impl_->expr};
  impl->expr = std::make_shared<CallExpr>(fn, args);
  return Access{std::move(impl)};
}

Access Access::operator+(const Access& rhs) const { return impl_->MakeCall("add", rhs); }
Access Access::operator*(const Access& rhs) const { return impl_->MakeCall("mul", rhs); }
Access Access::operator==(const Access& rhs) const { return impl_->MakeCall("eq", rhs); }

Access cond(const Access& lhs, const Access& rhs, const Access& true_case) {
  auto impl = std::make_unique<Access::Impl>();
  std::vector<std::shared_ptr<Expr>> args = {lhs.impl_->expr, rhs.impl_->expr, true_case.impl_->expr};
  impl->expr = std::make_shared<CallExpr>("cond", args);
  return Access{std::move(impl)};
}

const Access::Impl* Access::impl() const { return impl_.get(); }

class PolyEvaluator : public PolyVisitor {
 private:
  Polynomial Visit(const PolyIndex& expr) {
    auto it = seen_.find(expr.ptr);
    if (it == seen_.end()) {
      std::tie(it, std::ignore) = seen_.emplace(expr.ptr, NewIdx());
    }
    return Polynomial(it->second);
  }

  Polynomial Visit(const PolyLiteral& expr) { return Polynomial(expr.value); }

  Polynomial Visit(const PolyOp& expr) {
    if (expr.op == "neg") {
      return -expr.operands[0]->Accept(this);
    }
    if (expr.operands.size() != 2) {
      throw std::runtime_error("Invalid number of operands in PolyOp");
    }
    auto lhs = expr.operands[0]->Accept(this);
    auto rhs = expr.operands[1]->Accept(this);
    if (expr.op == "add") {
      return lhs + rhs;
    }
    if (expr.op == "sub") {
      return lhs - rhs;
    }
    if (expr.op == "mul") {
      if (lhs.isConstant()) {
        return rhs * lhs.constant();
      }
      if (rhs.isConstant()) {
        return lhs * rhs.constant();
      }
      throw std::runtime_error("Non-linear polynomial");
    }
    if (expr.op == "div") {
      if (!rhs.isConstant()) {
        throw std::runtime_error("Divisor of polynomials must be a constant");
      }
      return lhs / rhs.constant();
    }
    throw std::runtime_error("Unknown PolyOp");
  }

 private:
  std::string NewIdx() { return str(boost::format("x%1%") % next_++); }

 private:
  std::unordered_map<const void*, std::string> seen_;
  std::vector<Polynomial> constraints_;
  size_t next_ = 0;
};

class Evaluator : public AstVisitor<std::string> {
 public:
  Program Evaluate(const std::vector<Tensor>& vars) {
    for (const auto& var : vars) {
      auto expr = var.impl()->expr;
      auto it = seen_.find(expr.get());
      if (it == seen_.end()) {
        auto name = expr->Accept(this);
        std::tie(it, std::ignore) = seen_.emplace(expr.get(), name);
      }
      program_.outputs.push_back(it->second);
    }
    return program_;
  }

 private:
  std::string Visit(const ParamExpr& expr) {
    auto it = seen_.find(&expr);
    if (it == seen_.end()) {
      auto name = expr.name;
      if (name.empty()) {
        name = NewTmp();
      }
      tile::lang::Input input{tile::lang::Input::FIXED, name};
      for (size_t i = 0; i < expr.ndims; i++) {
        auto dim_name = str(boost::format("%1%_%2%") % name % i);
        input.dims.emplace_back(dim_name);
      }
      program_.inputs.push_back(input);
      std::tie(it, std::ignore) = seen_.emplace(&expr, name);
    }
    return it->second;
  }

  std::string Visit(const FloatConst& expr) {
    auto it = seen_.find(&expr);
    if (it == seen_.end()) {
      auto name = NewTmp();
      tile::lang::Op op{
          tile::lang::Op::CONSTANT,      // tag
          name,                          // output
          {std::to_string(expr.value)},  // inputs
          {},                            // Contraction
          {"fconst"},                    // Function
      };
      program_.ops.emplace_back(op);
      std::tie(it, std::ignore) = seen_.emplace(&expr, name);
    }
    return it->second;
  }

  std::string Visit(const IntConst& expr) {
    auto it = seen_.find(&expr);
    if (it == seen_.end()) {
      auto name = NewTmp();
      tile::lang::Op op{
          tile::lang::Op::CONSTANT,      // tag
          name,                          // output
          {std::to_string(expr.value)},  // inputs
          {},                            // Contraction
          {"iconst"},                    // Function
      };
      program_.ops.emplace_back(op);
      std::tie(it, std::ignore) = seen_.emplace(&expr, name);
    }
    return it->second;
  }

  std::string Visit(const ContractionExpr& expr) {
    auto it = seen_.find(&expr);
    if (it == seen_.end()) {
      PolyEvaluator poly_eval;
      tile::lang::Contraction cion;
      cion.agg_op = expr.agg_op;
      cion.comb_op = expr.combo_op;
      cion.no_defract = expr.no_defract;
      if (expr.use_default) {
        cion.use_default = expr.use_default->Accept(this);
      }
      cion.specs.emplace_back(tile::lang::TensorSpec{});
      std::vector<std::string> inputs;
      for (const auto& input : expr.inputs) {
        tile::lang::TensorSpec tensor_spec;
        tensor_spec.id = input->ref->Accept(this);
        inputs.push_back(tensor_spec.id);
        for (const auto& idx : input->index_spec) {
          auto poly = idx->Accept(&poly_eval);
          tensor_spec.spec.push_back(poly);
        }
        cion.specs.emplace_back(tensor_spec);
      }
      auto name = NewTmp();
      cion.specs[0].id = name;
      for (const auto& idx : expr.output->index_spec) {
        auto poly = idx->Accept(&poly_eval);
        cion.specs[0].spec.push_back(poly);
      }
      for (const auto& size : expr.output->output_sizes) {
        cion.output_size.push_back(std::to_string(size));
      }
      for (const auto& constraint : expr.constraints) {
        auto poly = constraint->lhs->Accept(&poly_eval);
        auto range = constraint->rhs;
        tile::math::RangeConstraint bound(poly, range);
        cion.constraints.emplace_back(bound);
      }
      tile::lang::Op op{
          tile::lang::Op::CONTRACTION,  // tag
          name,                         // output
          inputs,                       // inputs
          cion,                         // Contraction
      };
      program_.ops.emplace_back(op);
      std::tie(it, std::ignore) = seen_.emplace(&expr, name);
    }
    return it->second;
  }

  std::string Visit(const CallExpr& expr) {
    auto it = seen_.find(&expr);
    if (it == seen_.end()) {
      std::vector<std::string> args;
      for (const auto& arg : expr.args) {
        args.emplace_back(arg->Accept(this));
      }
      auto name = NewTmp();
      tile::lang::Op op{
          tile::lang::Op::FUNCTION,  // tag
          name,                      // output
          args,                      // inputs
          {},                        // Contraction
          {expr.fn},                 // Function
      };
      program_.ops.emplace_back(op);
      std::tie(it, std::ignore) = seen_.emplace(&expr, name);
    }
    return it->second;
  }

  std::string Visit(const TensorSpecExpr& expr) { throw std::runtime_error("Not implemented"); }
  std::string Visit(const ConstraintExpr& expr) { throw std::runtime_error("Not implemented"); }

 private:
  std::string NewTmp() { return str(boost::format("X%1%") % program_.next_tmp++); }

 private:
  std::unordered_map<const Expr*, std::string> seen_;
  Program program_;
};

bool MergeShapes(TensorShape* into, const TensorShape& shape) {
  IVLOG(4, "MergeShapes: " << *into << ", " << shape);
  if (shape.dims.size()) {
    if (into->dims.empty()) {
      into->dims = shape.dims;
      return false;
    } else if (into->dims != shape.dims) {
      IVLOG(4, "Checking compatibility between " << into->dims << " and " << shape.dims);
      auto dst = into->dims.rbegin();
      auto src = shape.dims.rbegin();
      for (;; ++dst, ++src) {
        IVLOG(4, "Top of check loop");
        if (src == shape.dims.rend()) {
          IVLOG(4, "src broadcasts to dst");
          break;
        }
        if (dst == into->dims.rend()) {
          // Anything that was used to produce 'into' can be broadcast
          // to 'src'; we just need to augment 'into' with the remaining
          // elements of 'src'.
          into->dims.insert(into->dims.begin(), shape.dims.begin(),
                            shape.dims.begin() + (shape.dims.size() - (src - shape.dims.rbegin())));
          IVLOG(4, "dst broadcasts to src; dims = " << into->dims);
          break;
        }
        IVLOG(4, "Considering " << dst->size << " vs. " << src->size);
        if (src->size == dst->size) {
          IVLOG(4, "No broadcasting needed (here)");
          continue;
        }
        if (src->size == 1) {
          // This dimension of src can be broadcast to whatever is in src.
          IVLOG(4, "dst broadcasts to src");
          continue;
        }
        if (dst->size == 1) {
          // This dimension of dst can be broadcast to whatever is in dst.
          dst->size = src->size;
          IVLOG(4, "src broadcasts to dst");
          continue;
        }
        // Otherwise, broadcasting cannot be done.
        throw std::runtime_error(
            str(boost::format("Mismatched tensor shapes in elementwise operation: %1% can't match %2%")  //
                % StreamContainer(into->dims) % StreamContainer(shape.dims)));
      }
      IVLOG(4, "Broadcast possible; LCM dims=" << into->dims);
      return true;
    }
  }
  return false;
}

TensorShape ComputeOutputShape(const std::vector<TensorShape>& inputs) {
  TensorShape ret;
  bool did_broadcast = false;
  for (const auto& input : inputs) {
    did_broadcast = MergeShapes(&ret, input) || did_broadcast;
  }
  if (did_broadcast) {
    // Recompute strides in dims.
    size_t stride = 1;
    for (auto it = ret.dims.rbegin(); it != ret.dims.rend(); ++it) {
      it->stride = stride;
      stride *= it->size;
    }
  }
  return ret;
}

Tensor Call(const std::string& fn, const std::vector<Tensor>& args) {
  auto impl = std::make_unique<Tensor::Impl>();
  std::vector<std::shared_ptr<Expr>> exprs;
  std::vector<TensorShape> shapes;
  for (const auto& tensor : args) {
    exprs.push_back(tensor.impl_->expr);
    shapes.push_back(tensor.impl_->shape);
  }
  impl->shape = ComputeOutputShape(shapes);
  impl->shape.type = tile::DataType::FLOAT32;  // FIXME: type promotion
  impl->expr = std::make_shared<CallExpr>(fn, exprs);
  return Tensor{std::move(impl)};
}

Tensor Tensor::reshape(const tile::TensorShape& shape) const {
  std::vector<Tensor> args = {*this};
  for (const auto& dim : shape.dims) {
    args.emplace_back(static_cast<int64_t>(dim.size));
  }
  auto ret = Call("reshape", args);
  ret.impl_->shape = shape;
  return ret;
}

Program Evaluate(const std::vector<Tensor>& vars) { return Evaluator().Evaluate(vars); }

}  // namespace tile_cc
}  // namespace plaidml
}  // namespace vertexai
