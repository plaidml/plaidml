// Copyright 2019 Intel Corporation.

#pragma once

#include <algorithm>
#include <functional>
#include <memory>
#include <ostream>
#include <sstream>
#include <string>
#include <tuple>
#include <utility>
#include <vector>

#include "plaidml2/core/core.h"
#include "plaidml2/edsl/ffi.h"

namespace plaidml {
namespace edsl {

class IndexedTensor;
class Tensor;
class TensorDim;
class TensorIndex;
class Value;

using TensorDeriv = std::vector<Tensor> (*)(  //
    const Tensor& Y,                          //
    const Tensor& dY,                         //
    const std::vector<Tensor>& Xs);

namespace details {

template <typename T>
struct Deleter {
  std::function<void(plaidml_error*, T*)> fn;
  void operator()(T* ptr) { ffi::call_void(fn, ptr); }
};

inline std::shared_ptr<plaidml_logical_shape> make_plaidml_logical_shape(plaidml_logical_shape* ptr) {
  return std::shared_ptr<plaidml_logical_shape>(ptr, Deleter<plaidml_logical_shape>{plaidml_logical_shape_free});
}

inline std::shared_ptr<plaidml_expr> make_plaidml_expr(plaidml_expr* ptr) {
  return std::shared_ptr<plaidml_expr>(ptr, Deleter<plaidml_expr>{plaidml_expr_free});
}

inline std::shared_ptr<plaidml_poly_expr> make_plaidml_poly_expr(plaidml_poly_expr* ptr) {
  return std::shared_ptr<plaidml_poly_expr>(ptr, Deleter<plaidml_poly_expr>{plaidml_poly_expr_free});
}

inline std::shared_ptr<plaidml_dim_expr> make_plaidml_dim_expr(plaidml_dim_expr* ptr) {
  return std::shared_ptr<plaidml_dim_expr>(ptr, Deleter<plaidml_dim_expr>{plaidml_dim_expr_free});
}

inline std::shared_ptr<plaidml_program> make_plaidml_program(plaidml_program* ptr) {
  return std::shared_ptr<plaidml_program>(ptr, Deleter<plaidml_program>{plaidml_program_free});
}

template <typename T>
void into_vector(std::vector<T>*) {}

template <typename T, typename Head, typename... Tail>
void into_vector(std::vector<T>* into, Head&& head, Tail&&... tail) {
  into->emplace_back(std::forward<Head>(head));
  into_vector(into, std::forward<Tail>(tail)...);
}

}  // namespace details

inline void init() {
  plaidml::init();
  ffi::call_void(plaidml_edsl_init);
}

class Program {
 public:
  Program(                                 //
      const std::string& name,             //
      const std::vector<Tensor>& outputs,  //
      const std::vector<std::tuple<Tensor, Tensor>>& updates = {});
  plaidml_program* as_ptr() const { return ptr_.get(); }
  std::string str() const { return ffi::str(ffi::call<plaidml_string*>(plaidml_program_repr, ptr_.get())); }
  const void* runinfo() const { return ffi::call<const void*>(plaidml_program_runinfo, ptr_.get()); }
  const std::vector<Tensor>& outputs() const { return outputs_; }

 private:
  std::shared_ptr<plaidml_program> ptr_;
  std::vector<Tensor> outputs_;
};

class TensorDim {
 public:
  TensorDim() : ptr_(details::make_plaidml_dim_expr(ffi::call<plaidml_dim_expr*>(plaidml_dim_expr_none))) {}

  explicit TensorDim(const std::shared_ptr<plaidml_dim_expr>& ptr) : ptr_(ptr) {}

  explicit TensorDim(int64_t value)
      : ptr_(details::make_plaidml_dim_expr(ffi::call<plaidml_dim_expr*>(plaidml_dim_expr_int, value))) {}

  TensorDim(plaidml_int_op op, const std::vector<TensorDim>& args)
      : ptr_(details::make_plaidml_dim_expr(MakeOp(op, args))) {}

  TensorDim operator-() const;

  std::string str() const {  //
    return ffi::str(ffi::call<plaidml_string*>(plaidml_dim_expr_repr, ptr_.get()));
  }

  int64_t as_int() const {
    if (!ptr_) {
      throw std::runtime_error("as_int() only available on TensorDim with an integer value");
    }
    return ffi::call<int64_t>(plaidml_dim_expr_get_int, ptr_.get());
  }

  plaidml_dim_expr* as_ptr() const { return ptr_.get(); }

 private:
  static plaidml_dim_expr* MakeOp(plaidml_int_op op, const std::vector<TensorDim>& args) {
    std::vector<plaidml_dim_expr*> operands;
    for (const auto& arg : args) {
      operands.push_back(arg.as_ptr());
    }
    return ffi::call<plaidml_dim_expr*>(plaidml_dim_expr_op, op, operands.size(), operands.data());
  }

 private:
  std::shared_ptr<plaidml_dim_expr> ptr_;
};

struct Constraint;

class TensorIndex {
 public:
  TensorIndex() : ptr_(details::make_plaidml_poly_expr(ffi::call<plaidml_poly_expr*>(plaidml_poly_expr_index, ""))) {}

  explicit TensorIndex(int64_t value)
      : ptr_(details::make_plaidml_poly_expr(ffi::call<plaidml_poly_expr*>(plaidml_poly_expr_literal, value))) {}

  explicit TensorIndex(const std::string& name)
      : ptr_(details::make_plaidml_poly_expr(ffi::call<plaidml_poly_expr*>(plaidml_poly_expr_index, name.c_str()))) {}

  TensorIndex(plaidml_int_op op, const std::vector<TensorIndex>& args)
      : ptr_(details::make_plaidml_poly_expr(MakePolyOp(op, args))) {}

  TensorIndex(plaidml_int_op op, const TensorIndex& idx, const TensorDim& dim, bool lhs_first)
      : ptr_(details::make_plaidml_poly_expr(MakeDimPolyOp(op, idx, dim, lhs_first))) {}

  TensorIndex operator-() const;

  Constraint operator<(int64_t rhs) const;

  Constraint operator<(const TensorDim& rhs) const;

  std::string str() const {  //
    return ffi::str(ffi::call<plaidml_string*>(plaidml_poly_expr_repr, as_ptr()));
  }

  plaidml_poly_expr* as_ptr() const { return ptr_.get(); }

 private:
  static plaidml_poly_expr* MakePolyOp(plaidml_int_op op, const std::vector<TensorIndex>& args) {
    std::vector<plaidml_poly_expr*> operands;
    for (const auto& arg : args) {
      operands.push_back(arg.as_ptr());
    }
    return ffi::call<plaidml_poly_expr*>(plaidml_poly_expr_op, op, operands.size(), operands.data());
  }

  static plaidml_poly_expr* MakeDimPolyOp(plaidml_int_op op, const TensorIndex& idx, const TensorDim& dim,
                                          bool lhs_first) {
    std::vector<plaidml_poly_expr*> operands;
    auto dim_ptr = ffi::call<plaidml_poly_expr*>(plaidml_poly_expr_dim, dim.as_ptr());
    if (lhs_first) {
      operands.emplace_back(idx.as_ptr());
      operands.emplace_back(dim_ptr);
    } else {
      operands.emplace_back(dim_ptr);
      operands.emplace_back(idx.as_ptr());
    }
    return ffi::call<plaidml_poly_expr*>(plaidml_poly_expr_op, op, operands.size(), operands.data());
  }

 private:
  std::shared_ptr<plaidml_poly_expr> ptr_;
};

struct Constraint {
  TensorIndex lhs;
  TensorDim rhs;
};

class IndexedTensor {
  friend class Tensor;

  struct ComboParts {
    plaidml_combo_op op;
    std::vector<plaidml_expr*> args;
  };

  struct Impl {
    std::shared_ptr<plaidml_expr> idxs;
    std::shared_ptr<plaidml_expr> sizes;
    std::shared_ptr<ComboParts> rhs;
    const Tensor* src = nullptr;
    void MakeContraction(plaidml_agg_op agg_op, const IndexedTensor& rhs);
  };

 public:
  ~IndexedTensor() = default;

  IndexedTensor(plaidml_combo_op op, const std::vector<const IndexedTensor*>& args) : impl_(new Impl()) {
    impl_->rhs = std::make_shared<ComboParts>();
    impl_->rhs->op = op;
    for (const auto& arg : args) {
      impl_->rhs->args.emplace_back(arg->impl_->idxs.get());
    }
  }

  // Movable constructor
  IndexedTensor(IndexedTensor&& rhs) noexcept : impl_(std::move(rhs.impl_)) {}

  // Represents an aggregation_op of SUM in a contraction
  IndexedTensor& operator+=(const IndexedTensor& rhs) {
    impl_->MakeContraction(PLAIDML_AGG_OP_SUM, rhs);
    return *this;
  }

  // Represents an aggregation_op of PROD in a contraction
  IndexedTensor& operator*=(const IndexedTensor& rhs) {
    impl_->MakeContraction(PLAIDML_AGG_OP_PROD, rhs);
    return *this;
  }

  // Represents an aggregation_op of MAX in a contraction
  IndexedTensor& operator>=(const IndexedTensor& rhs) {
    impl_->MakeContraction(PLAIDML_AGG_OP_MAX, rhs);
    return *this;
  }

  // Represents an aggregation_op of MIN in a contraction
  IndexedTensor& operator<=(const IndexedTensor& rhs) {
    impl_->MakeContraction(PLAIDML_AGG_OP_MIN, rhs);
    return *this;
  }

  // Represents an aggregation_op of ASSIGN in a contraction
  IndexedTensor& operator=(const IndexedTensor& rhs) {
    impl_->MakeContraction(PLAIDML_AGG_OP_ASSIGN, rhs);
    return *this;
  }

  // Represents a combo_op of PLUS in a contraction
  IndexedTensor operator+(const IndexedTensor& rhs) const;

  // Represents a combo_op of MULTIPLY in a contraction
  IndexedTensor operator*(const IndexedTensor& rhs) const;

  // Represents a combo_op of EQ in a contraction
  IndexedTensor operator==(const IndexedTensor& rhs) const;

 private:
  std::unique_ptr<Impl> impl_;
  explicit IndexedTensor(std::unique_ptr<Impl> impl) : impl_(std::move(impl)) {}
};

inline IndexedTensor sum(IndexedTensor lhs, const IndexedTensor& rhs) { return std::move(lhs += rhs); }

inline IndexedTensor product(IndexedTensor lhs, const IndexedTensor& rhs) { return std::move(lhs *= rhs); }

inline IndexedTensor max(IndexedTensor lhs, const IndexedTensor& rhs) { return std::move(lhs >= rhs); }

inline IndexedTensor min(IndexedTensor lhs, const IndexedTensor& rhs) { return std::move(lhs <= rhs); }

inline IndexedTensor assign(IndexedTensor lhs, const IndexedTensor& rhs) { return std::move(lhs = rhs); }

class LogicalShape {
  friend class Tensor;

 public:
  LogicalShape(plaidml_datatype dtype, const std::vector<int64_t>& dims)
      : ptr_(details::make_plaidml_logical_shape(
            ffi::call<plaidml_logical_shape*>(plaidml_logical_shape_alloc, dtype, dims.size(), dims.data()))) {}

  plaidml_logical_shape* as_ptr() const { return ptr_.get(); }

  std::string str() const {  //
    return ffi::str(ffi::call<plaidml_string*>(plaidml_logical_shape_repr, ptr_.get()));
  }

  plaidml_datatype dtype() const {  //
    return ffi::call<plaidml_datatype>(plaidml_logical_shape_get_dtype, ptr_.get());
  }

  size_t ndims() const {  //
    return ffi::call<size_t>(plaidml_logical_shape_get_ndims, ptr_.get());
  }

  std::vector<int64_t> int_dims() const {
    std::vector<int64_t> ret(ndims());
    for (size_t i = 0; i < ret.size(); i++) {
      ret[i] = ffi::call<int64_t>(plaidml_logical_shape_get_dim_int, ptr_.get(), i);
    }
    return ret;
  }

  // std::vector<TensorDim> dims() const {
  //   std::vector<TensorDim> ret(ndims());
  //   for (size_t i = 0; i < ret.size(); i++) {
  //     auto impl = std::make_shared<TensorDim::Impl>();
  //     impl->ptr = details::make_plaidml_dim_expr(
  //         ffi::call<plaidml_dim_expr*>(plaidml_logical_shape_get_dim_expr, ptr_.get(), i));
  //     ret[i] = TensorDim(impl);
  //   }
  //   return ret;
  // }

  bool operator==(const LogicalShape& rhs) const { return str() == rhs.str(); }

 private:
  explicit LogicalShape(const std::shared_ptr<plaidml_logical_shape>& ptr) : ptr_(ptr) {}

 private:
  std::shared_ptr<plaidml_logical_shape> ptr_;
};

class Tensor {
  friend class IndexedTensor;
  friend class Value;

  struct Impl {
    std::shared_ptr<plaidml_expr> ptr;
    bool has_dims = false;
    std::vector<TensorDim> dims;
    std::string name;
  };

 public:
  Tensor() : impl_(new Impl) {  //
    impl_->ptr = details::make_plaidml_expr(ffi::call<plaidml_expr*>(plaidml_expr_none));
  }

  explicit Tensor(plaidml_expr* ptr) : impl_(new Impl) {  //
    impl_->ptr = details::make_plaidml_expr(ptr);
  }

  explicit Tensor(int value) : impl_(new Impl) {  //
    impl_->ptr = details::make_plaidml_expr(ffi::call<plaidml_expr*>(plaidml_expr_int, value));
  }

  explicit Tensor(size_t value) : impl_(new Impl) {  //
    impl_->ptr = details::make_plaidml_expr(ffi::call<plaidml_expr*>(plaidml_expr_int, value));
  }

  explicit Tensor(int64_t value) : impl_(new Impl) {
    impl_->ptr = details::make_plaidml_expr(ffi::call<plaidml_expr*>(plaidml_expr_int, value));
  }

  explicit Tensor(double value) : impl_(new Impl) {
    impl_->ptr = details::make_plaidml_expr(ffi::call<plaidml_expr*>(plaidml_expr_float, value));
  }

  explicit Tensor(const TensorDim& dim) : impl_(new Impl) {
    impl_->ptr = details::make_plaidml_expr(ffi::call<plaidml_expr*>(plaidml_expr_dim, dim.as_ptr()));
  }

  explicit Tensor(const std::vector<int64_t>& dims) : impl_(new Impl) {
    for (auto dim : dims) {
      impl_->dims.emplace_back(dim);
    }
    impl_->has_dims = true;
  }

  explicit Tensor(const std::vector<TensorDim>& dims) : impl_(new Impl) {
    impl_->dims = dims;
    impl_->has_dims = true;
  }

  explicit Tensor(const std::initializer_list<TensorDim>& dims) : impl_(new Impl) {
    impl_->dims = dims;
    impl_->has_dims = true;
  }

  Tensor(const std::string& name, const std::vector<TensorDim>& dims) : impl_(new Impl) {
    impl_->name = name;
    impl_->dims = dims;
    impl_->has_dims = true;
  }

  Tensor(const std::string& name, const std::initializer_list<TensorDim>& dims) : impl_(new Impl) {
    impl_->name = name;
    impl_->dims = dims;
    impl_->has_dims = true;
  }

  // Copyable
  Tensor(const Tensor& rhs) { *this = rhs; }

  Tensor& operator=(const Tensor& rhs) {
    if (this != &rhs) {
      impl_.reset(new Impl(*rhs.impl_));
    }
    return *this;
  }

  IndexedTensor operator()(const std::vector<TensorIndex>& idxs) const {
    std::vector<plaidml_poly_expr*> idx_ptrs(idxs.size());
    for (size_t i = 0; i < idxs.size(); i++) {
      idx_ptrs[i] = idxs[i].as_ptr();
    }
    std::unique_ptr<IndexedTensor::Impl> impl(new IndexedTensor::Impl());
    impl->src = this;
    if (impl_->has_dims) {
      std::vector<plaidml_dim_expr*> sizes;
      for (const auto& dim : impl_->dims) {
        sizes.emplace_back(dim.as_ptr());
      }
      impl->sizes = details::make_plaidml_expr(  //
          ffi::call<plaidml_expr*>(              //
              plaidml_expr_size_map,             //
              sizes.size(),                      //
              sizes.data()));
    }
    impl->idxs = details::make_plaidml_expr(  //
        ffi::call<plaidml_expr*>(             //
            plaidml_expr_index_map,           //
            as_ptr(),                         //
            idx_ptrs.size(),                  //
            idx_ptrs.data()));
    return IndexedTensor{std::move(impl)};
  }

  IndexedTensor operator()(const std::initializer_list<TensorIndex>& idxs) const {
    return operator()(std::vector<TensorIndex>{idxs});
  }

  template <typename... Ts>
  IndexedTensor operator()(Ts... idxs) const {
    std::vector<TensorIndex> vec;
    details::into_vector(&vec, std::forward<Ts>(idxs)...);
    return operator()(vec);
  }

  // Represents an eltwise negation
  Tensor operator-() const;

  // Represents an eltwise bit_not
  Tensor operator~() const;

  std::string str() const {  //
    return ffi::str(ffi::call<plaidml_string*>(plaidml_expr_repr, as_ptr()));
  }

  plaidml_expr* as_ptr() const { return impl_->ptr.get(); }

  // Enable no_reduce on a contraction
  Tensor& no_reduce() {
    ffi::call_void(plaidml_expr_contraction_set_no_reduce, as_ptr(), true);
    return *this;
  }

  // Set use_default on a contraction
  Tensor& use_default(const Tensor& rhs) {
    ffi::call_void(plaidml_expr_contraction_set_use_default, as_ptr(), rhs.as_ptr());
    return *this;
  }

  void add_constraint(const Constraint& constraint) {
    ffi::call_void(plaidml_expr_contraction_add_constraint, as_ptr(), constraint.lhs.as_ptr(), constraint.rhs.as_ptr());
  }

  void add_constraints(const std::vector<Constraint>& constraints) {
    for (const auto& constraint : constraints) {
      add_constraint(constraint);
    }
  }

  // Return the tensor's shape
  LogicalShape shape() const {
    auto ptr = details::make_plaidml_logical_shape(ffi::call<plaidml_logical_shape*>(plaidml_expr_get_shape, as_ptr()));
    return LogicalShape(ptr);
  }

  // Verify that the specified dims match the dims of this tensor.
  void bind_dims(const std::vector<TensorDim>& dims) const {
    std::vector<plaidml_dim_expr*> raw_dims(dims.size());
    for (size_t i = 0; i < dims.size(); i++) {
      raw_dims[i] = dims[i].as_ptr();
    }
    ffi::call_void(plaidml_expr_bind_dims, as_ptr(), raw_dims.size(), raw_dims.data());
  }

  template <typename... Ts>
  void bind_dims(Ts... dims) const {
    std::vector<TensorDim> vec;
    details::into_vector(&vec, std::forward<Ts>(dims)...);
    bind_dims(vec);
  }

 private:
  explicit Tensor(std::unique_ptr<Impl> impl) : impl_(std::move(impl)) {}

 private:
  std::unique_ptr<Impl> impl_;
};

template <typename... Ts>
Tensor NamedTensorOutput(const std::string& name, Ts&&... dims) {
  std::vector<TensorDim> vec;
  details::into_vector(&vec, std::forward<Ts>(dims)...);
  return Tensor{name, vec};
}

inline Tensor NamedTensorOutput(const std::string& name, const std::vector<TensorDim>& dims) {  //
  return Tensor{name, dims};
}

template <typename... Ts>
Tensor TensorOutput(Ts... dims) {
  std::vector<TensorDim> vec;
  details::into_vector(&vec, std::forward<Ts>(dims)...);
  return Tensor{vec};
}

inline Tensor TensorOutput(const std::vector<TensorDim>& dims) {  //
  return Tensor(dims);
}

inline Tensor TensorOutput(const std::vector<int64_t>& dims) {  //
  return Tensor(dims);
}

inline Tensor Placeholder(      //
    const LogicalShape& shape,  //
    const std::string& name = "") {
  auto ptr = ffi::call<plaidml_expr*>(  //
      plaidml_expr_placeholder,         //
      shape.as_ptr(),                   //
      nullptr,                          //
      name.c_str());
  return Tensor(ptr);
}

inline Tensor Placeholder(             //
    plaidml_datatype dtype,            //
    const std::vector<int64_t>& dims,  //
    const std::string& name = "") {
  LogicalShape shape(dtype, dims);
  return Placeholder(shape, name);
}

inline plaidml_deriv ThunkTensorDeriv(TensorDeriv fn) {
  return [](void* user_ctx,          //
            plaidml_expr* Y_expr,    //
            plaidml_expr* dY_expr,   //
            size_t nXs,              //
            plaidml_expr** X_exprs,  //
            plaidml_expr** dX_exprs) {
    auto fn = reinterpret_cast<TensorDeriv>(user_ctx);
    Tensor Y(Y_expr);
    Tensor dY(dY_expr);
    std::vector<Tensor> Xs(nXs);
    for (size_t i = 0; i < Xs.size(); i++) {
      Xs[i] = Tensor(X_exprs[i]);
    }
    auto dXs = fn(Y, dY, Xs);
    for (size_t i = 0; i < Xs.size(); i++) {
      dX_exprs[i] = ffi::call<plaidml_expr*>(plaidml_expr_clone, dXs[i].as_ptr());
    }
  };
}

inline Tensor OverrideGrads(TensorDeriv fn, const std::vector<Tensor>& ins, const Tensor& out) {
  auto thunk = ThunkTensorDeriv(fn);
  auto nins = ins.size();
  std::vector<plaidml_expr*> in_ptrs(nins);
  for (size_t i = 0; i < ins.size(); i++) {
    in_ptrs[i] = ins[i].as_ptr();
  }
  auto ptr = ffi::call<plaidml_expr*>(plaidml_expr_grad_override, thunk, reinterpret_cast<void*>(fn), nins,
                                      in_ptrs.data(), out.as_ptr());
  return Tensor(ptr);
}

Tensor Call(const std::string& fn, const std::vector<Tensor>& args);

template <typename... Ts>
Tensor Call(const std::string& fn, Ts... args) {
  std::vector<Tensor> vec;
  details::into_vector(&vec, std::forward<Ts>(args)...);
  return Call(fn, vec);
}

inline Tensor cast(const Tensor& x, plaidml_datatype dtype) {
  auto ptr = ffi::call<plaidml_expr*>(plaidml_expr_cast, x.as_ptr(), dtype);
  return Tensor{ptr};
}

inline Tensor abs(const Tensor& x) { return Call("abs", x); }

inline Tensor as_float(const Tensor& x, size_t bit_size) {
  switch (bit_size) {
    case 16:
      return cast(x, PLAIDML_DATA_FLOAT16);
    case 32:
      return cast(x, PLAIDML_DATA_FLOAT32);
    case 64:
      return cast(x, PLAIDML_DATA_FLOAT64);
    default:
      throw std::runtime_error("Invalid bit size for as_float");
  }
}

inline Tensor as_int(const Tensor& x, size_t bit_size) {
  switch (bit_size) {
    case 8:
      return cast(x, PLAIDML_DATA_INT8);
    case 16:
      return cast(x, PLAIDML_DATA_INT16);
    case 32:
      return cast(x, PLAIDML_DATA_INT32);
    case 64:
      return cast(x, PLAIDML_DATA_INT64);
    default:
      throw std::runtime_error("Invalid bit size for as_int");
  }
}

inline Tensor as_uint(const Tensor& x, size_t bit_size) {
  switch (bit_size) {
    case 8:
      return cast(x, PLAIDML_DATA_UINT8);
    case 16:
      return cast(x, PLAIDML_DATA_UINT16);
    case 32:
      return cast(x, PLAIDML_DATA_UINT32);
    case 64:
      return cast(x, PLAIDML_DATA_UINT64);
    default:
      throw std::runtime_error("Invalid bit size for as_uint");
  }
}

inline Tensor as_bool(const Tensor& x) { return cast(x, PLAIDML_DATA_BOOLEAN); }

inline Tensor cos(const Tensor& x) { return Call("cos", x); }

inline Tensor cosh(const Tensor& x) { return Call("cosh", x); }

inline Tensor exp(const Tensor& x) { return Call("exp", x); }

inline Tensor gather(const Tensor& x, const Tensor& y) { return Call("gather", x, y); }

inline Tensor ident(const Tensor& x) { return Call("ident", x); }

inline Tensor index(const Tensor& x, size_t axis) { return Call("index", x, static_cast<int64_t>(axis)); }

inline Tensor log(const Tensor& x) { return Call("log", x); }

inline Tensor pow(const Tensor& x, const Tensor& y) { return Call("pow", x, y); }

inline Tensor prng_state(const Tensor& x) { return Call("prng_state", x); }

inline Tensor prng_step(const Tensor& x, const std::vector<size_t>& sizes) {
  std::vector<Tensor> args = {x};
  for (const auto& size : sizes) {
    args.emplace_back(static_cast<int64_t>(size));
  }
  return Call("prng_step", args);
}

inline Tensor prng_value(const Tensor& x) { return Call("prng_value", x); }

inline Tensor reshape(const Tensor& x, const std::vector<int64_t>& dims) {
  std::vector<Tensor> args = {x};
  for (const auto& dim : dims) {
    args.emplace_back(dim);
  }
  return Call("reshape", args);
}

inline Tensor reshape(const Tensor& x, const std::vector<TensorDim>& dims) {
  std::vector<Tensor> args = {x};
  for (const auto& dim : dims) {
    args.emplace_back(dim);
  }
  return Call("reshape", args);
}

inline Tensor scatter(const Tensor& x, const Tensor& y, const Tensor& z) { return Call("scatter", x, y, z); }

inline Tensor select(const Tensor& cond, const Tensor& true_case, const Tensor& false_case) {
  return Call("cond", cond, true_case, false_case);
}

inline Tensor shape(const Tensor& x) { return Call("shape", x); }

inline Tensor sin(const Tensor& x) { return Call("sin", x); }

inline Tensor sinh(const Tensor& x) { return Call("sinh", x); }

inline Tensor sqrt(const Tensor& x) { return Call("sqrt", x); }

inline Tensor tan(const Tensor& x) { return Call("tan", x); }

inline Tensor tanh(const Tensor& x) { return Call("tanh", x); }

inline Tensor zero() { return Tensor{0}; }

inline Program::Program(                 //
    const std::string& name,             //
    const std::vector<Tensor>& outputs,  //
    const std::vector<std::tuple<Tensor, Tensor>>& updates)
    : outputs_(outputs.size()) {
  std::vector<plaidml_expr*> raw_outputs(outputs.size());
  std::vector<plaidml_expr*> new_outputs(outputs.size());
  for (size_t i = 0; i < raw_outputs.size(); i++) {
    auto ptr = outputs[i].as_ptr();
    if (!ptr) {
      std::stringstream ss;
      ss << "Invalid tensor output requested by Program: " << outputs[i].str();
      throw std::runtime_error(ss.str());
    }
    raw_outputs[i] = ptr;
  }
  std::vector<plaidml_expr*> src_updates;
  std::vector<plaidml_expr*> dst_updates;
  for (size_t i = 0; i < updates.size(); i++) {
    dst_updates[i] = std::get<0>(updates[i]).as_ptr();
    src_updates[i] = std::get<1>(updates[i]).as_ptr();
  }
  ptr_ = details::make_plaidml_program(ffi::call<plaidml_program*>(  //
      plaidml_program_evaluate,                                      //
      name.c_str(),                                                  //
      raw_outputs.size(),                                            //
      raw_outputs.data(),                                            //
      new_outputs.data(),                                            //
      updates.size(),                                                //
      src_updates.data(),                                            //
      dst_updates.data()));
  for (size_t i = 0; i < new_outputs.size(); i++) {
    outputs_[i] = Tensor(new_outputs[i]);
  }
}

inline TensorDim TensorDim::operator-() const { return TensorDim(PLAIDML_INT_OP_NEG, {*this}); }

inline TensorIndex TensorIndex::operator-() const { return TensorIndex(PLAIDML_INT_OP_NEG, {*this}); }

inline Constraint TensorIndex::operator<(int64_t rhs) const { return Constraint{*this, TensorDim(rhs)}; }

inline Constraint TensorIndex::operator<(const TensorDim& rhs) const { return Constraint{*this, rhs}; }

#define PLAIDML_EDSL_DEFINE_TENSOR_IDXDIM_BINARY_OPS(_op_, _int_op_, _fn_)           \
  inline TensorIndex operator _op_(const TensorIndex& lhs, const TensorIndex& rhs) { \
    return TensorIndex(_int_op_, {lhs, rhs});                                        \
  }                                                                                  \
  inline TensorIndex operator _op_(const TensorIndex& lhs, int64_t rhs) {            \
    return TensorIndex(_int_op_, {lhs, TensorIndex{rhs}});                           \
  }                                                                                  \
  inline TensorIndex operator _op_(int64_t lhs, const TensorIndex& rhs) {            \
    return TensorIndex(_int_op_, {TensorIndex{lhs}, rhs});                           \
  }                                                                                  \
  inline TensorIndex operator _op_(const TensorIndex& lhs, const TensorDim& rhs) {   \
    return TensorIndex(_int_op_, lhs, rhs, true);                                    \
  }                                                                                  \
  inline TensorIndex operator _op_(const TensorDim& lhs, const TensorIndex& rhs) {   \
    return TensorIndex(_int_op_, rhs, lhs, false);                                   \
  }                                                                                  \
  inline Tensor operator _op_(const Tensor& lhs, const TensorDim& rhs) { /**/        \
    return Call(_fn_, lhs, Tensor(rhs));                                             \
  }                                                                                  \
  inline Tensor operator _op_(const TensorDim& lhs, const Tensor& rhs) { /**/        \
    return Call(_fn_, Tensor(lhs), rhs);                                             \
  }                                                                                  \
  inline TensorDim operator _op_(const TensorDim& lhs, const TensorDim& rhs) {       \
    return TensorDim(_int_op_, {lhs, rhs});                                          \
  }                                                                                  \
  inline TensorDim operator _op_(int64_t lhs, const TensorDim& rhs) {                \
    return TensorDim(_int_op_, {TensorDim{lhs}, rhs});                               \
  }                                                                                  \
  inline TensorDim operator _op_(const TensorDim& lhs, int64_t rhs) {                \
    return TensorDim(_int_op_, {lhs, TensorDim{rhs}});                               \
  }

PLAIDML_EDSL_DEFINE_TENSOR_IDXDIM_BINARY_OPS(+, PLAIDML_INT_OP_ADD, "add");
PLAIDML_EDSL_DEFINE_TENSOR_IDXDIM_BINARY_OPS(-, PLAIDML_INT_OP_SUB, "sub");
PLAIDML_EDSL_DEFINE_TENSOR_IDXDIM_BINARY_OPS(*, PLAIDML_INT_OP_MUL, "mul");
PLAIDML_EDSL_DEFINE_TENSOR_IDXDIM_BINARY_OPS(/, PLAIDML_INT_OP_DIV, "div");

#define PLAIDML_EDSL_DEFINE_TENSOR_DIM_BINARY_FN(_fn_, _int_op_)                                                  \
  inline TensorDim _fn_(const TensorDim& lhs, const TensorDim& rhs) { return TensorDim{_int_op_, {lhs, rhs}}; }   \
  inline TensorDim _fn_(int64_t lhs, const TensorDim& rhs) { return TensorDim{_int_op_, {TensorDim{lhs}, rhs}}; } \
  inline TensorDim _fn_(const TensorDim& lhs, int64_t rhs) { return TensorDim{_int_op_, {lhs, TensorDim{rhs}}}; }

PLAIDML_EDSL_DEFINE_TENSOR_DIM_BINARY_FN(max, PLAIDML_INT_OP_MAX);
PLAIDML_EDSL_DEFINE_TENSOR_DIM_BINARY_FN(min, PLAIDML_INT_OP_MIN);

inline Tensor Tensor::operator-() const { return Call("neg", {*this}); }
inline Tensor Tensor::operator~() const { return Call("bit_not", {*this}); }

#define PLAIDML_EDSL_DEFINE_TENSOR_BINARY_OPS(_op_, _fn_)                                              \
  inline Tensor operator _op_(const Tensor& lhs, const Tensor& rhs) { return Call(_fn_, lhs, rhs); }   \
  inline Tensor operator _op_(const Tensor& lhs, int rhs) { return Call(_fn_, lhs, Tensor{rhs}); }     \
  inline Tensor operator _op_(const Tensor& lhs, int64_t rhs) { return Call(_fn_, lhs, Tensor{rhs}); } \
  inline Tensor operator _op_(const Tensor& lhs, double rhs) { return Call(_fn_, lhs, Tensor{rhs}); }  \
  inline Tensor operator _op_(int lhs, const Tensor& rhs) { return Call(_fn_, Tensor{lhs}, rhs); }     \
  inline Tensor operator _op_(int64_t lhs, const Tensor& rhs) { return Call(_fn_, Tensor{lhs}, rhs); } \
  inline Tensor operator _op_(double lhs, const Tensor& rhs) { return Call(_fn_, Tensor{lhs}, rhs); }

PLAIDML_EDSL_DEFINE_TENSOR_BINARY_OPS(+, "add");
PLAIDML_EDSL_DEFINE_TENSOR_BINARY_OPS(-, "sub");
PLAIDML_EDSL_DEFINE_TENSOR_BINARY_OPS(*, "mul");
PLAIDML_EDSL_DEFINE_TENSOR_BINARY_OPS(/, "div");
PLAIDML_EDSL_DEFINE_TENSOR_BINARY_OPS(==, "cmp_eq");
PLAIDML_EDSL_DEFINE_TENSOR_BINARY_OPS(!=, "cmp_ne");
PLAIDML_EDSL_DEFINE_TENSOR_BINARY_OPS(<, "cmp_lt");
PLAIDML_EDSL_DEFINE_TENSOR_BINARY_OPS(>, "cmp_gt");
PLAIDML_EDSL_DEFINE_TENSOR_BINARY_OPS(<=, "cmp_le");
PLAIDML_EDSL_DEFINE_TENSOR_BINARY_OPS(>=, "cmp_ge");
PLAIDML_EDSL_DEFINE_TENSOR_BINARY_OPS(<<, "bit_left");
PLAIDML_EDSL_DEFINE_TENSOR_BINARY_OPS(>>, "bit_right");
PLAIDML_EDSL_DEFINE_TENSOR_BINARY_OPS(&, "bit_and");
PLAIDML_EDSL_DEFINE_TENSOR_BINARY_OPS(|, "bit_or");
PLAIDML_EDSL_DEFINE_TENSOR_BINARY_OPS(^, "bit_xor");

inline void IndexedTensor::Impl::MakeContraction(plaidml_agg_op agg_op, const IndexedTensor& rhs) {
  plaidml_combo_op combo_op;
  std::vector<plaidml_expr*> src_idxs;
  if (rhs.impl_->rhs) {
    // n-ary op: ComboParts
    combo_op = rhs.impl_->rhs->op;
    src_idxs = rhs.impl_->rhs->args;
  } else {
    // unary op: TensorSpec
    combo_op = PLAIDML_COMBO_OP_NONE;
    src_idxs.emplace_back(rhs.impl_->idxs.get());
  }
  src->impl_->ptr = details::make_plaidml_expr(  //
      ffi::call<plaidml_expr*>(                  //
          plaidml_expr_contraction,              //
          agg_op,                                //
          combo_op,                              //
          idxs.get(),                            //
          sizes.get(),                           //
          src_idxs.size(),                       //
          src_idxs.data(),                       //
          src->impl_->name.c_str()));
}

// Represents a combo_op of COND in a contraction
inline IndexedTensor cond(const IndexedTensor& lhs, const IndexedTensor& rhs, const IndexedTensor& true_case) {
  return IndexedTensor(PLAIDML_COMBO_OP_COND, {&lhs, &rhs, &true_case});
}

inline IndexedTensor IndexedTensor::operator+(const IndexedTensor& rhs) const {  //
  return IndexedTensor(PLAIDML_COMBO_OP_ADD, {this, &rhs});
}

inline IndexedTensor IndexedTensor::operator*(const IndexedTensor& rhs) const {  //
  return IndexedTensor(PLAIDML_COMBO_OP_MUL, {this, &rhs});
}

inline IndexedTensor IndexedTensor::operator==(const IndexedTensor& rhs) const {  //
  return IndexedTensor(PLAIDML_COMBO_OP_EQ, {this, &rhs});
}

inline Tensor Call(const std::string& fn, const std::vector<Tensor>& args) {
  std::vector<plaidml_expr*> ptrs(args.size());
  for (size_t i = 0; i < args.size(); i++) {
    ptrs[i] = args[i].as_ptr();
  }
  auto ptr = ffi::call<plaidml_expr*>(  //
      plaidml_expr_call,                //
      fn.c_str(),                       //
      ptrs.size(),                      //
      ptrs.data());
  return Tensor{ptr};
}

class Value {
 public:
  Value() : ptr_(details::make_plaidml_expr(ffi::call<plaidml_expr*>(plaidml_expr_none))) {}

  explicit Value(plaidml_expr* ptr)
      : ptr_(details::make_plaidml_expr(ffi::call<plaidml_expr*>(plaidml_expr_clone, ptr))) {}

  explicit Value(int value) : ptr_(details::make_plaidml_expr(ffi::call<plaidml_expr*>(plaidml_expr_int, value))) {}

  explicit Value(size_t value) : ptr_(details::make_plaidml_expr(ffi::call<plaidml_expr*>(plaidml_expr_int, value))) {}

  explicit Value(int64_t value) : ptr_(details::make_plaidml_expr(ffi::call<plaidml_expr*>(plaidml_expr_int, value))) {}

  explicit Value(double value)
      : ptr_(details::make_plaidml_expr(ffi::call<plaidml_expr*>(plaidml_expr_float, value))) {}

  explicit Value(const std::string& value)
      : ptr_(details::make_plaidml_expr(ffi::call<plaidml_expr*>(plaidml_expr_str, value.c_str()))) {}

  explicit Value(const TensorDim& dim)
      : ptr_(details::make_plaidml_expr(ffi::call<plaidml_expr*>(plaidml_expr_dim, dim.as_ptr()))) {}

  explicit Value(const Tensor& tensor)
      : ptr_(details::make_plaidml_expr(ffi::call<plaidml_expr*>(plaidml_expr_clone, tensor.as_ptr()))) {}

  explicit Value(const std::vector<Value>& tuple) {
    std::vector<plaidml_expr*> args(tuple.size());
    for (size_t i = 0; i < args.size(); i++) {
      args[i] = tuple[i].as_ptr();
    }
    ptr_ = details::make_plaidml_expr(ffi::call<plaidml_expr*>(plaidml_expr_tuple, args.size(), args.data()));
  }

  std::string str() const {  //
    return ffi::str(ffi::call<plaidml_string*>(plaidml_expr_repr, as_ptr()));
  }

  bool is_none() const {  //
    return ffi::call<plaidml_expr_kind>(plaidml_expr_get_kind, as_ptr()) == PLAIDML_EXPR_NONE;
  }

  bool is_int() const {  //
    return ffi::call<plaidml_expr_kind>(plaidml_expr_get_kind, as_ptr()) == PLAIDML_EXPR_INT;
  }

  bool is_float() const {  //
    return ffi::call<plaidml_expr_kind>(plaidml_expr_get_kind, as_ptr()) == PLAIDML_EXPR_FLOAT;
  }

  bool is_tensor() const {
    return ffi::call<plaidml_expr_kind>(plaidml_expr_get_kind, as_ptr()) == PLAIDML_EXPR_TENSOR;
  }

  bool is_tuple() const {  //
    return ffi::call<plaidml_expr_kind>(plaidml_expr_get_kind, as_ptr()) == PLAIDML_EXPR_TUPLE;
  }

  bool is_str() const {  //
    return ffi::call<plaidml_expr_kind>(plaidml_expr_get_kind, as_ptr()) == PLAIDML_EXPR_STR;
  }

  bool is_dim() const {  //
    return ffi::call<plaidml_expr_kind>(plaidml_expr_get_kind, as_ptr()) == PLAIDML_EXPR_DIM;
  }

  bool as_bool() const {
    // bools are integers under the hood, but we can still return a bool type
    return static_cast<bool>(ffi::call<int64_t>(plaidml_expr_int_get_value, as_ptr()));
  }

  int64_t as_int() const {  //
    return ffi::call<int64_t>(plaidml_expr_int_get_value, as_ptr());
  }

  double as_float() const {  //
    return ffi::call<double>(plaidml_expr_float_get_value, as_ptr());
  }

  std::string as_str() const {  //
    return ffi::str(ffi::call<plaidml_string*>(plaidml_expr_str_get_value, as_ptr()));
  }

  Tensor as_tensor() const {  //
    return Tensor(ffi::call<plaidml_expr*>(plaidml_expr_clone, as_ptr()));
  }

  TensorDim as_dim() const {  //
    return TensorDim(details::make_plaidml_dim_expr(ffi::call<plaidml_dim_expr*>(plaidml_expr_get_dim, as_ptr())));
  }

  std::vector<Value> as_tuple() const {
    auto count = ffi::call<size_t>(plaidml_expr_tuple_get_count, as_ptr());
    std::vector<Value> ret(count);
    std::vector<plaidml_expr*> exprs(count);
    ffi::call_void(plaidml_expr_tuple_get_exprs, as_ptr(), exprs.size(), exprs.data());
    for (size_t i = 0; i < ret.size(); i++) {
      ret[i] = Value{exprs[i]};
    }
    return ret;
  }

  std::vector<int64_t> as_int_tuple() const {
    auto count = ffi::call<size_t>(plaidml_expr_tuple_get_count, as_ptr());
    std::vector<int64_t> ret(count);
    std::vector<plaidml_expr*> exprs(count);
    ffi::call_void(plaidml_expr_tuple_get_exprs, as_ptr(), exprs.size(), exprs.data());
    for (size_t i = 0; i < ret.size(); i++) {
      ret[i] = Value{exprs[i]}.as_int();
    }
    return ret;
  }

  plaidml_expr* as_ptr() const { return ptr_.get(); }

 private:
  std::shared_ptr<plaidml_expr> ptr_;
};

template <typename... Ts>
Value make_tuple(Ts... elts) {
  std::vector<Value> vec;
  details::into_vector(&vec, std::forward<Ts>(elts)...);
  return Value{vec};
}

template <typename T>
Value make_tuple(const std::vector<T>& elts) {
  std::vector<Value> vec(elts.size());
  for (size_t i = 0; i < vec.size(); i++) {
    vec[i] = Value{elts[i]};
  }
  return Value{vec};
}

inline Value make_tuple(const std::vector<Value>& elts) {  //
  return Value{elts};
}

inline Value None() { return Value(); }

inline std::ostream& operator<<(std::ostream& os, const LogicalShape& x) {
  os << x.str();
  return os;
}

inline std::ostream& operator<<(std::ostream& os, const Tensor& x) {
  os << x.str();
  return os;
}

inline std::ostream& operator<<(std::ostream& os, const TensorDim& x) {
  os << x.str();
  return os;
}

inline std::ostream& operator<<(std::ostream& os, const TensorIndex& x) {
  os << x.str();
  return os;
}

inline std::ostream& operator<<(std::ostream& os, const Program& x) {
  os << x.str();
  return os;
}

inline std::ostream& operator<<(std::ostream& os, const Value& x) {
  os << x.str();
  return os;
}

}  // namespace edsl
}  // namespace plaidml
