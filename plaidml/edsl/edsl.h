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

#include "plaidml/core/core.h"
#include "plaidml/edsl/ffi.h"

namespace plaidml {
namespace edsl {

class IndexedTensor;
class Program;
struct ProgramArgument;
class Tensor;
class TensorDim;
class TensorIndex;
struct TensorRef;
class Value;

using TensorDeriv = std::vector<Tensor> (*)(  //
    const Tensor& Y,                          //
    const Tensor& dY,                         //
    const std::vector<Tensor>& Xs);

namespace details {

struct Deleter {
  void operator()(plaidml_dim_expr* ptr) { ffi::call_void(plaidml_dim_expr_free, ptr); }
  void operator()(plaidml_expr* ptr) { ffi::call_void(plaidml_expr_free, ptr); }
  void operator()(plaidml_logical_shape* ptr) { ffi::call_void(plaidml_logical_shape_free, ptr); }
  void operator()(plaidml_poly_expr* ptr) { ffi::call_void(plaidml_poly_expr_free, ptr); }
  void operator()(plaidml_program* ptr) { ffi::call_void(plaidml_program_free, ptr); }
  void operator()(plaidml_tuple* ptr) { ffi::call_void(plaidml_tuple_free, ptr); }
  void operator()(plaidml_value* ptr) { ffi::call_void(plaidml_value_free, ptr); }
};

template <typename T>
inline std::shared_ptr<T> make_ptr(T* ptr) {
  return std::shared_ptr<T>(ptr, Deleter{});
}

template <typename T>
void into_vector(std::vector<T>*) {}

template <typename T, typename Head, typename... Tail>
void into_vector(std::vector<T>* into, Head&& head, Tail&&... tail) {
  into->emplace_back(std::forward<Head>(head));
  into_vector(into, std::forward<Tail>(tail)...);
}

}  // namespace details

///
/// Initializes PlaidML's EDSL API.
///
inline void init() {
  plaidml::init();
  ffi::call_void(plaidml_edsl_init);
}

///
/// \defgroup edsl_objects Objects
///

///
/// \ingroup edsl_objects
/// \class Program
/// This is a program.
///
class Program {
 public:
  ///
  /// Program constructor
  ///
  Program(                                 //
      const std::string& name,             //
      const std::vector<Tensor>& outputs,  //
      const std::vector<std::tuple<Tensor, Tensor>>& updates = {});

  ///
  /// Return the Program as a string.
  ///
  std::string str() const { return ffi::str(ffi::call<plaidml_string*>(plaidml_program_repr, ptr_.get())); }

  ///
  /// args
  ///
  const std::vector<ProgramArgument>& args() const { return args_; }

  ///
  /// inputs
  ///
  const std::vector<ProgramArgument>& inputs() const { return inputs_; }

  ///
  /// outputs
  ///
  const std::vector<ProgramArgument>& outputs() const { return outputs_; }

  plaidml_program* as_ptr() const { return ptr_.get(); }

 private:
  std::shared_ptr<plaidml_program> ptr_;
  std::vector<ProgramArgument> args_;
  std::vector<ProgramArgument> inputs_;
  std::vector<ProgramArgument> outputs_;
};

///
/// \ingroup edsl_objects
/// \class TensorDim
/// A symbolic object used to specify the dimensions of a Tensor
///
class TensorDim {
 public:
  ///
  /// TensorDim constructor
  ///
  TensorDim() : ptr_(details::make_ptr(ffi::call<plaidml_dim_expr*>(plaidml_dim_expr_none))) {}

  ///
  /// TensorDim constructor
  ///
  explicit TensorDim(const std::shared_ptr<plaidml_dim_expr>& ptr) : ptr_(ptr) {}

  ///
  /// TensorDim constructor
  ///
  explicit TensorDim(int64_t value)
      : ptr_(details::make_ptr(ffi::call<plaidml_dim_expr*>(plaidml_dim_expr_int, value))) {}

  TensorDim(plaidml_int_op op, const std::vector<TensorDim>& args) : ptr_(details::make_ptr(MakeOp(op, args))) {}

  ///
  /// Represents a subtraction operator overload.
  ///
  TensorDim operator-() const;

  ///
  /// Returns the TensorDim as a string.
  ///
  std::string str() const {  //
    return ffi::str(ffi::call<plaidml_string*>(plaidml_dim_expr_repr, ptr_.get()));
  }

  ///
  /// Returns the TensorDim as an int.
  ///
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

///
/// \ingroup edsl_objects
/// \class TensorIndex
/// A symbolic object used to directly index a Tensor or to compute a Tensor's index as part of a formula.
///
class TensorIndex {
 public:
  ///
  /// TensorIndex constructor
  ///
  TensorIndex() : ptr_(details::make_ptr(ffi::call<plaidml_poly_expr*>(plaidml_poly_expr_index, ""))) {}

  ///
  /// TensorIndex constructor
  ///
  explicit TensorIndex(int64_t value)
      : ptr_(details::make_ptr(ffi::call<plaidml_poly_expr*>(plaidml_poly_expr_literal, value))) {}

  ///
  /// TensorIndex constructor
  ///
  explicit TensorIndex(const std::string& name)
      : ptr_(details::make_ptr(ffi::call<plaidml_poly_expr*>(plaidml_poly_expr_index, name.c_str()))) {}

  TensorIndex(plaidml_int_op op, const std::vector<TensorIndex>& args)
      : ptr_(details::make_ptr(MakePolyOp(op, args))) {}

  TensorIndex(plaidml_int_op op, const TensorIndex& idx, const TensorDim& dim, bool lhs_first)
      : ptr_(details::make_ptr(MakeDimPolyOp(op, idx, dim, lhs_first))) {}

  ///
  /// Represents an subtraction operator overload on a TensorIndex
  ///
  TensorIndex operator-() const;

  ///
  /// TODO
  ///
  Constraint operator<(int64_t rhs) const;

  ///
  /// TODO
  ///
  Constraint operator<(const TensorDim& rhs) const;

  ///
  /// Returns the TensorIndex as a string.
  ///
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

///
/// \ingroup edsl_objects
/// \struct Constraint
/// This is a constraint.
///
struct Constraint {
  ///
  /// lhs
  ///
  TensorIndex lhs;

  ///
  /// rhs
  ///
  TensorDim rhs;
};

///
/// \ingroup edsl_objects
/// \class IndexedTensor
/// This is an IndexedTensor
///
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

  ///
  /// Represents an aggregation_op of SUM in a contraction
  ///
  IndexedTensor& operator+=(const IndexedTensor& rhs) {
    impl_->MakeContraction(PLAIDML_AGG_OP_SUM, rhs);
    return *this;
  }

  ///
  /// Represents an aggregation_op of PROD in a contraction
  ///
  IndexedTensor& operator*=(const IndexedTensor& rhs) {
    impl_->MakeContraction(PLAIDML_AGG_OP_PROD, rhs);
    return *this;
  }

  ///
  /// Represents an aggregation_op of MAX in a contraction
  ///
  IndexedTensor& operator>=(const IndexedTensor& rhs) {
    impl_->MakeContraction(PLAIDML_AGG_OP_MAX, rhs);
    return *this;
  }

  ///
  /// Represents an aggregation_op of MIN in a contraction
  ///
  IndexedTensor& operator<=(const IndexedTensor& rhs) {
    impl_->MakeContraction(PLAIDML_AGG_OP_MIN, rhs);
    return *this;
  }

  ///
  /// Represents an aggregation_op of ASSIGN in a contraction
  ///
  IndexedTensor& operator=(const IndexedTensor& rhs) {
    impl_->MakeContraction(PLAIDML_AGG_OP_ASSIGN, rhs);
    return *this;
  }

  ///
  /// Represents a combo_op of PLUS in a contraction
  ///
  IndexedTensor operator+(const IndexedTensor& rhs) const;

  ///
  /// Represents a combo_op of MULTIPLY in a contraction
  ///
  IndexedTensor operator*(const IndexedTensor& rhs) const;

  ///
  /// Represents a combo_op of EQ in a contraction
  ///
  IndexedTensor operator==(const IndexedTensor& rhs) const;

 private:
  std::unique_ptr<Impl> impl_;
  explicit IndexedTensor(std::unique_ptr<Impl> impl) : impl_(std::move(impl)) {}
};

///
/// \ingroup edsl_objects
/// \class LogicalShape
/// This is a LogicalShape.
///
class LogicalShape {
  friend class Program;
  friend class Tensor;

 public:
  ///
  /// LogicalShape constructor
  ///
  LogicalShape(DType dtype, const std::vector<int64_t>& dims)
      : ptr_(details::make_ptr(ffi::call<plaidml_logical_shape*>(
            plaidml_logical_shape_alloc, static_cast<plaidml_datatype>(dtype), dims.size(), dims.data()))) {}

  ///
  /// Returns a LogicalShape as a string
  ///
  std::string str() const {  //
    return ffi::str(ffi::call<plaidml_string*>(plaidml_logical_shape_repr, ptr_.get()));
  }

  ///
  /// Returns the datatype of the LogicalShape
  ///
  DType dtype() const {
    auto ret = ffi::call<plaidml_datatype>(plaidml_logical_shape_get_dtype, ptr_.get());
    return static_cast<DType>(ret);
  }

  ///
  /// Returns the number of dimensions of the LogicalShape
  ///
  size_t ndims() const {  //
    return ffi::call<size_t>(plaidml_logical_shape_get_ndims, ptr_.get());
  }

  ///
  /// Returns the dimensions of the LogicalShape as a vector of integers.
  ///
  std::vector<int64_t> int_dims() const {
    std::vector<int64_t> ret(ndims());
    for (size_t i = 0; i < ret.size(); i++) {
      ret[i] = ffi::call<int64_t>(plaidml_logical_shape_get_dim_int, ptr_.get(), i);
    }
    return ret;
  }

  ///
  /// TODO
  ///
  bool operator==(const LogicalShape& rhs) const { return str() == rhs.str(); }

  plaidml_logical_shape* as_ptr() const { return ptr_.get(); }

 private:
  explicit LogicalShape(plaidml_logical_shape* ptr) : ptr_(details::make_ptr(ptr)) {}

 private:
  std::shared_ptr<plaidml_logical_shape> ptr_;
};

///
/// \ingroup edsl_objects
/// \class Tensor
/// A multidimensional array of a fixed shape.
///
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
  ///
  /// Tensor constructor
  ///
  Tensor() : impl_(new Impl) {}

  explicit Tensor(plaidml_expr* ptr) : impl_(new Impl) {  //
    impl_->ptr = details::make_ptr(ptr);
  }

  ///
  /// Tensor constructor
  /// \param value int
  /// \return Tensor
  ///
  explicit Tensor(int value) : impl_(new Impl) {  //
    impl_->ptr = details::make_ptr(ffi::call<plaidml_expr*>(plaidml_expr_int, value));
  }

  ///
  /// Tensor constructor
  /// \param value unsigned int
  /// \return Tensor
  ///
  explicit Tensor(unsigned value) : impl_(new Impl) {  //
    impl_->ptr = details::make_ptr(ffi::call<plaidml_expr*>(plaidml_expr_uint, value));
  }

  ///
  /// Tensor constructor
  /// \param value unsigned 64-bit int
  /// \return Tensor
  ///
  explicit Tensor(uint64_t value) : impl_(new Impl) {  //
    impl_->ptr = details::make_ptr(ffi::call<plaidml_expr*>(plaidml_expr_uint, value));
  }

  ///
  /// Tensor constructor
  /// \param value int64_t
  /// \return Tensor
  ///
  explicit Tensor(int64_t value) : impl_(new Impl) {
    impl_->ptr = details::make_ptr(ffi::call<plaidml_expr*>(plaidml_expr_int, value));
  }

  ///
  /// Tensor constructor
  /// \param value double
  /// \return Tensor
  ///
  explicit Tensor(double value) : impl_(new Impl) {
    impl_->ptr = details::make_ptr(ffi::call<plaidml_expr*>(plaidml_expr_float, value));
  }

  ///
  /// Tensor constructor
  ///
  explicit Tensor(const TensorDim& dim) : impl_(new Impl) {
    impl_->ptr = details::make_ptr(ffi::call<plaidml_expr*>(plaidml_expr_dim, dim.as_ptr()));
  }

  ///
  /// Tensor constructor
  ///
  explicit Tensor(const std::vector<int64_t>& dims) : impl_(new Impl) {
    for (auto dim : dims) {
      impl_->dims.emplace_back(dim);
    }
    impl_->has_dims = true;
  }

  ///
  /// Tensor constructor
  ///
  explicit Tensor(const std::vector<TensorDim>& dims) : impl_(new Impl) {
    impl_->dims = dims;
    impl_->has_dims = true;
  }

  ///
  /// Tensor constructor
  ///
  explicit Tensor(const std::initializer_list<TensorDim>& dims) : impl_(new Impl) {
    impl_->dims = dims;
    impl_->has_dims = true;
  }

  ///
  /// Tensor constructor
  ///
  Tensor(const std::string& name, const std::vector<TensorDim>& dims) : impl_(new Impl) {
    impl_->name = name;
    impl_->dims = dims;
    impl_->has_dims = true;
  }

  ///
  /// Tensor constructor
  ///
  Tensor(const std::string& name, const std::initializer_list<TensorDim>& dims) : impl_(new Impl) {
    impl_->name = name;
    impl_->dims = dims;
    impl_->has_dims = true;
  }

  ///
  /// Tensor constructor
  ///
  Tensor(const Tensor& rhs) { *this = rhs; }

  ///
  /// Represents an operator overload for `=` for a `Tensor`
  ///
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
      impl->sizes = details::make_ptr(  //
          ffi::call<plaidml_expr*>(     //
              plaidml_expr_size_map,    //
              sizes.size(),             //
              sizes.data()));
    }
    impl->idxs = details::make_ptr(  //
        ffi::call<plaidml_expr*>(    //
            plaidml_expr_index_map,  //
            as_ptr(),                //
            idx_ptrs.size(),         //
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

  ///
  /// Represents an eltwise negation
  ///
  Tensor operator-() const;

  ///
  /// Represents an eltwise bit_not
  ///
  Tensor operator~() const;

  ///
  /// TODO
  ///
  std::string str() const {  //
    return ffi::str(ffi::call<plaidml_string*>(plaidml_expr_repr, as_ptr()));
  }

  ///
  /// Enable no_reduce on a contraction
  ///
  Tensor& no_reduce() {
    ffi::call_void(plaidml_expr_contraction_set_no_reduce, as_ptr(), true);
    return *this;
  }

  ///
  /// Set use_default on a contraction
  ///
  Tensor& use_default(const Tensor& rhs) {
    ffi::call_void(plaidml_expr_contraction_set_use_default, as_ptr(), rhs.as_ptr());
    return *this;
  }

  ///
  /// TODO
  ///
  Tensor& add_constraint(const Constraint& constraint) {
    ffi::call_void(plaidml_expr_contraction_add_constraint, as_ptr(), constraint.lhs.as_ptr(), constraint.rhs.as_ptr());
    return *this;
  }

  ///
  /// TODO
  ///
  Tensor& add_constraints(const std::vector<Constraint>& constraints) {
    for (const auto& constraint : constraints) {
      add_constraint(constraint);
    }
    return *this;
  }

  ///
  /// Return the tensor's shape
  ///
  LogicalShape shape() const {
    return LogicalShape(ffi::call<plaidml_logical_shape*>(plaidml_expr_get_shape, as_ptr()));
  }

  ///
  /// Verify that the specified dims match the dims of this tensor.
  ///
  void bind_dims(const std::vector<TensorDim>& dims) const {
    std::vector<plaidml_dim_expr*> raw_dims(dims.size());
    for (size_t i = 0; i < dims.size(); i++) {
      raw_dims[i] = dims[i].as_ptr();
    }
    ffi::call_void(plaidml_expr_bind_dims, as_ptr(), raw_dims.size(), raw_dims.data());
  }

  ///
  /// TODO
  ///
  template <typename... Ts>
  void bind_dims(Ts... dims) const {
    std::vector<TensorDim> vec;
    details::into_vector(&vec, std::forward<Ts>(dims)...);
    bind_dims(vec);
  }

  plaidml_expr* as_ptr() const { return impl_->ptr.get(); }

  void* raw_ptr() const { return ffi::call<void*>(plaidml_expr_ptr, as_ptr()); }

 private:
  explicit Tensor(std::unique_ptr<Impl> impl) : impl_(std::move(impl)) {}

 private:
  std::unique_ptr<Impl> impl_;
};

///
/// \ingroup edsl_objects
/// \struct TensorRef
/// A reference to a Tensor
///
struct TensorRef {
  ///
  /// The `Tensor` that the `TensorRef` is referencing
  ///
  Tensor tensor;

  ///
  /// TensorRef constructor
  ///
  TensorRef(const Tensor& tensor) : tensor(tensor) {}  // NOLINT[runtime/explicit]

  ///
  /// TODO
  ///
  operator Tensor() const { return tensor; }

  ///
  /// TODO
  ///
  bool operator<(const TensorRef& rhs) const { return tensor.raw_ptr() < rhs.tensor.raw_ptr(); }

  ///
  /// TODO
  ///
  bool operator==(const TensorRef& rhs) const { return tensor.raw_ptr() == rhs.tensor.raw_ptr(); }
};

///
/// \ingroup edsl_objects
/// \struct ProgramArgument
/// Description for ProgramArgument
///
struct ProgramArgument {
  ///
  /// TODO
  ///
  bool is_input;

  ///
  /// TODO
  ///
  TensorRef tensor;

  ///
  /// TODO
  ///
  LogicalShape shape;

  ///
  /// TODO
  ///
  std::shared_ptr<Buffer> buffer;
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
    DType dtype,                       //
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

///
/// \defgroup edsl_primitives EDSL Primitives
///

/// \addtogroup edsl_primitives
/// @{

///
/// Computes the elementwise absolute value of `x`.
/// \param x Tensor
/// \return Tensor
///
inline Tensor abs(const Tensor& x) { return Call("abs", x); }

///
/// Casts the element type of a tensor `x` to the type specified by `dtype`.
/// \param x Tensor
/// \param dtype DType
/// \return Tensor
///
inline Tensor cast(const Tensor& x, DType dtype) {
  auto ptr = ffi::call<plaidml_expr*>(plaidml_expr_cast, x.as_ptr(), static_cast<plaidml_datatype>(dtype));
  return Tensor{ptr};
}

///
/// Computes the elementwise ceiling of `x`.
/// \param x Tensor
/// \return Tensor
///
inline Tensor ceil(const Tensor& x) { return Call("ceil", x); }

///
/// Computes the elementwise cosine of `x`.
/// \param x Tensor
/// \return Tensor
///
inline Tensor cos(const Tensor& x) { return Call("cos", x); }

///
/// Computes the elementwise hyperbolic cosine of `x`.
/// \param x Tensor
/// \return Tensor
///
inline Tensor cosh(const Tensor& x) { return Call("cosh", x); }

///
/// Computes the elementwise natural exponential function of `x`: _e_<sup>x</sup>.
/// \param x Tensor
/// \return Tensor
///
inline Tensor exp(const Tensor& x) { return Call("exp", x); }

///
/// Computes the elementwise floor of `x`.
/// \param x Tensor
/// \return Tensor
///
inline Tensor floor(const Tensor& x) { return Call("floor", x); }

///
/// Takes an input tensor (`x`) and a set of indices to gather over (`y`), and returns an output tensor that gathers the
/// input tensor from the indices specified.
/// \param x Tensor
/// \param y Tensor
/// \return Tensor
///
inline Tensor gather(const Tensor& x, const Tensor& y) { return Call("gather", x, y); }

///
/// Returns the identity of `x`.
/// \param x Tensor
/// \return Tensor
///
inline Tensor ident(const Tensor& x) { return Call("ident", x); }

///
/// Returns the index of `x` at the specified `axis`.
/// \param x Tensor
/// \param axis size_t
/// \return Tensor
///
inline Tensor index(const Tensor& x, size_t axis) { return Call("index", x, static_cast<int64_t>(axis)); }

///
/// Computes the elementwise natural logarithm of `x`: _ln_(`x`).
/// \param x Tensor
/// \return Tensor
///
inline Tensor log(const Tensor& x) { return Call("log", x); }

///
/// Computes the elementwise `y`th power of `x`.
/// \param x Tensor
/// \param y Tensor
/// \return Tensor
///
inline Tensor pow(const Tensor& x, const Tensor& y) { return Call("pow", x, y); }

///
/// Generates a Tensor of elementwise pseudorandom numbers using the seed values specified in `state`.
/// \param state Tensor
/// \param dims vector<int64_t>
/// \return Tensor
///
inline Tensor prng(const Tensor& state, const std::vector<int64_t>& dims) {
  std::vector<Tensor> args = {state};
  for (const auto& dim : dims) {
    args.emplace_back(dim);
  }
  return Call("prng", args);
}

///
/// Takes an input tensor `x` and reshapes it according to `dims`.
/// \param x Tensor
/// \param dims vector<int64_t>
/// \return Tensor
///
inline Tensor reshape(const Tensor& x, const std::vector<int64_t>& dims) {
  std::vector<Tensor> args = {x};
  for (const auto& dim : dims) {
    args.emplace_back(dim);
  }
  return Call("reshape", args);
}

///
/// Takes an input tensor `x` and reshapes it according to `dims`.
/// \param x Tensor
/// \param dims vector<TensorDim>
/// \return Tensor
///
inline Tensor reshape(const Tensor& x, const std::vector<TensorDim>& dims) {
  std::vector<Tensor> args = {x};
  for (const auto& dim : dims) {
    args.emplace_back(dim);
  }
  return Call("reshape", args);
}

///
/// Rounds `x` elementwise.
/// \param x Tensor
/// \return Tensor
///
inline Tensor round(const Tensor& x) { return Call("round", x); }

///
/// Takes an input tensor (`x`), a set of indices to scatter over (`y`), and the number of elements in the scattered
/// tensor (`z`), and returns an output tensor that scatters the input tensor across the number of elements specified.
/// \param x Tensor
/// \param y Tensor
/// \param z Tensor
/// \return Tensor
///
inline Tensor scatter(const Tensor& x, const Tensor& y, const Tensor& z) { return Call("scatter", x, y, z); }

///
/// Performs an elementwise conditional which returns the corresponding
/// element in `true_case` if the condition is evaluated to be true or the
/// corresponding element in `false_case` if the condition is evaluated to be
/// false.
/// \param cond Tensor
/// \param true_case Tensor
/// \param false_case Tensor
/// \return Tensor
///
inline Tensor select(const Tensor& cond, const Tensor& true_case, const Tensor& false_case) {
  return Call("cond", cond, true_case, false_case);
}

///
/// Returns the shape of `x` as a Tensor.
/// \param x Tensor
/// \return Tensor
///
inline Tensor shape(const Tensor& x) { return Call("shape", x); }

///
/// Computes the elementwise sine of `x`.
/// \param x Tensor
/// \return Tensor
///
inline Tensor sin(const Tensor& x) { return Call("sin", x); }

///
/// Computes the elementwise hyperbolic sine of `x`.
/// \param x Tensor
/// \return Tensor
///
inline Tensor sinh(const Tensor& x) { return Call("sinh", x); }

///
/// Computes the elementwise square root of `x`.
/// \param x Tensor
/// \return Tensor
///
inline Tensor sqrt(const Tensor& x) { return Call("sqrt", x); }

///
/// Computes the elementwise tangent of `x`.
/// \param x Tensor
/// \return Tensor
///
inline Tensor tan(const Tensor& x) { return Call("tan", x); }

///
/// Computes the elementwise hyperbolic tangent of `x`.
/// \param x Tensor
/// \return Tensor
///
inline Tensor tanh(const Tensor& x) { return Call("tanh", x); }

///
/// Returns a Tensor with a value of 0.
/// \return Tensor
///
inline Tensor zero() { return Tensor{0}; }

/// @}

inline Program::Program(                 //
    const std::string& name,             //
    const std::vector<Tensor>& outputs,  //
    const std::vector<std::tuple<Tensor, Tensor>>& updates) {
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
  plaidml_program_args* args;
  ptr_ = details::make_ptr(ffi::call<plaidml_program*>(  //
      plaidml_program_evaluate,                          //
      name.c_str(),                                      //
      raw_outputs.size(),                                //
      raw_outputs.data(),                                //
      updates.size(),                                    //
      src_updates.data(),                                //
      dst_updates.data(),                                //
      &args));
  for (size_t i = 0; i < args->nargs; i++) {
    const auto& arg = args->args[i];
    Tensor tensor(ffi::call<plaidml_expr*>(plaidml_expr_clone, arg.tensor));
    LogicalShape shape(ffi::call<plaidml_logical_shape*>(plaidml_logical_shape_clone, arg.shape));
    ProgramArgument programArg{arg.is_input, tensor, shape};
    if (arg.buffer) {
      TensorShape tensor_shape(shape.dtype(), shape.int_dims());
      auto bufptr = ffi::call<plaidml_buffer*>(plaidml_buffer_clone, arg.buffer);
      programArg.buffer = std::make_shared<Buffer>(bufptr, tensor_shape);
    }
    if (arg.is_input) {
      inputs_.push_back(programArg);
    } else {
      outputs_.push_back(programArg);
    }
    args_.emplace_back(programArg);
  }
  ffi::call_void(plaidml_program_args_free, args);
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

#define PLAIDML_EDSL_DEFINE_TENSOR_BINARY_OPS(_op_, _fn_)                                               \
  inline Tensor operator _op_(const Tensor& lhs, const Tensor& rhs) { return Call(_fn_, lhs, rhs); }    \
  inline Tensor operator _op_(const Tensor& lhs, int rhs) { return Call(_fn_, lhs, Tensor{rhs}); }      \
  inline Tensor operator _op_(const Tensor& lhs, int64_t rhs) { return Call(_fn_, lhs, Tensor{rhs}); }  \
  inline Tensor operator _op_(const Tensor& lhs, uint64_t rhs) { return Call(_fn_, lhs, Tensor{rhs}); } \
  inline Tensor operator _op_(const Tensor& lhs, double rhs) { return Call(_fn_, lhs, Tensor{rhs}); }   \
  inline Tensor operator _op_(int lhs, const Tensor& rhs) { return Call(_fn_, Tensor{lhs}, rhs); }      \
  inline Tensor operator _op_(int64_t lhs, const Tensor& rhs) { return Call(_fn_, Tensor{lhs}, rhs); }  \
  inline Tensor operator _op_(uint64_t lhs, const Tensor& rhs) { return Call(_fn_, Tensor{lhs}, rhs); } \
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
PLAIDML_EDSL_DEFINE_TENSOR_BINARY_OPS(<<, "bit_shl");
PLAIDML_EDSL_DEFINE_TENSOR_BINARY_OPS(>>, "bit_shr");
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
  src->impl_->ptr = details::make_ptr(  //
      ffi::call<plaidml_expr*>(         //
          plaidml_expr_contraction,     //
          agg_op,                       //
          combo_op,                     //
          idxs.get(),                   //
          sizes.get(),                  //
          src_idxs.size(),              //
          src_idxs.data(),              //
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
  Value() : ptr_(details::make_ptr(ffi::call<plaidml_value*>(plaidml_value_none))) {}

  explicit Value(plaidml_value* ptr) : ptr_(details::make_ptr(ffi::call<plaidml_value*>(plaidml_value_clone, ptr))) {}

  explicit Value(int value) : ptr_(details::make_ptr(ffi::call<plaidml_value*>(plaidml_value_int, value))) {}

  explicit Value(size_t value) : ptr_(details::make_ptr(ffi::call<plaidml_value*>(plaidml_value_int, value))) {}

  explicit Value(int64_t value) : ptr_(details::make_ptr(ffi::call<plaidml_value*>(plaidml_value_int, value))) {}

  explicit Value(double value) : ptr_(details::make_ptr(ffi::call<plaidml_value*>(plaidml_value_double, value))) {}

  explicit Value(float value) : ptr_(details::make_ptr(ffi::call<plaidml_value*>(plaidml_value_float, value))) {}

  explicit Value(const std::string& value)
      : ptr_(details::make_ptr(ffi::call<plaidml_value*>(plaidml_value_str, value.c_str()))) {}

  explicit Value(const TensorDim& dim)
      : ptr_(details::make_ptr(ffi::call<plaidml_value*>(plaidml_value_dim, dim.as_ptr()))) {}

  explicit Value(const Tensor& tensor) {
    if (auto ptr = tensor.as_ptr()) {
      ptr_ = details::make_ptr(ffi::call<plaidml_value*>(plaidml_value_expr, ptr));
    } else {
      ptr_ = details::make_ptr(ffi::call<plaidml_value*>(plaidml_value_none));
    }
  }

  explicit Value(const std::vector<Value>& tuple) {
    std::vector<plaidml_value*> args(tuple.size());
    for (size_t i = 0; i < args.size(); i++) {
      args[i] = tuple[i].as_ptr();
    }
    ptr_ = details::make_ptr(ffi::call<plaidml_value*>(plaidml_value_tuple, args.size(), args.data()));
  }

  std::string str() const {  //
    return ffi::str(ffi::call<plaidml_string*>(plaidml_value_repr, as_ptr()));
  }

  bool is_none() const {  //
    return ffi::call<plaidml_value_kind>(plaidml_value_get_kind, as_ptr()) == PLAIDML_VALUE_NONE;
  }

  bool is_int() const {  //
    return ffi::call<plaidml_value_kind>(plaidml_value_get_kind, as_ptr()) == PLAIDML_VALUE_INT;
  }

  bool is_double() const {  //
    return ffi::call<plaidml_value_kind>(plaidml_value_get_kind, as_ptr()) == PLAIDML_VALUE_DOUBLE;
  }

  bool is_float() const {  //
    return ffi::call<plaidml_value_kind>(plaidml_value_get_kind, as_ptr()) == PLAIDML_VALUE_FLOAT;
  }

  bool is_tensor() const {
    return ffi::call<plaidml_value_kind>(plaidml_value_get_kind, as_ptr()) == PLAIDML_VALUE_EXPR;
  }

  bool is_tuple() const {  //
    return ffi::call<plaidml_value_kind>(plaidml_value_get_kind, as_ptr()) == PLAIDML_VALUE_TUPLE;
  }

  bool is_str() const {  //
    return ffi::call<plaidml_value_kind>(plaidml_value_get_kind, as_ptr()) == PLAIDML_VALUE_STR;
  }

  bool is_dim() const {  //
    return ffi::call<plaidml_value_kind>(plaidml_value_get_kind, as_ptr()) == PLAIDML_VALUE_DIM;
  }

  bool as_bool() const {
    // bools are integers under the hood, but we can still return a bool type
    return static_cast<bool>(ffi::call<int64_t>(plaidml_value_int_get, as_ptr()));
  }

  int64_t as_int() const {  //
    return ffi::call<int64_t>(plaidml_value_int_get, as_ptr());
  }

  float as_float() const {  //
    return ffi::call<float>(plaidml_value_float_get, as_ptr());
  }

  double as_double() const {  //
    return ffi::call<double>(plaidml_value_float_get, as_ptr());
  }

  std::string as_str() const {  //
    return ffi::str(ffi::call<plaidml_string*>(plaidml_value_str_get, as_ptr()));
  }

  Tensor as_tensor() const {
    if (is_tensor()) {
      return Tensor(ffi::call<plaidml_expr*>(plaidml_value_expr_get, as_ptr()));
    }
    if (is_dim()) {
      return Tensor(as_dim());
    }
    if (is_double()) {
      return Tensor(as_double());
    }
    if (is_float()) {
      return Tensor(as_float());
    }
    if (is_int()) {
      return Tensor(as_int());
    }
    throw std::runtime_error("Value cannot be coerced into Tensor");
  }

  TensorDim as_dim() const {  //
    return TensorDim(details::make_ptr(ffi::call<plaidml_dim_expr*>(plaidml_value_dim_get, as_ptr())));
  }

  std::vector<Value> as_tuple() const {
    auto tuple = details::make_ptr(ffi::call<plaidml_tuple*>(plaidml_value_tuple_get, as_ptr()));
    std::vector<Value> ret(tuple->nelts);
    for (size_t i = 0; i < ret.size(); i++) {
      ret[i] = Value{tuple->elts[i]};
    }
    return ret;
  }

  std::vector<int64_t> as_int_tuple() const {
    auto tuple = details::make_ptr(ffi::call<plaidml_tuple*>(plaidml_value_tuple_get, as_ptr()));
    std::vector<int64_t> ret(tuple->nelts);
    for (size_t i = 0; i < ret.size(); i++) {
      ret[i] = Value{tuple->elts[i]}.as_int();
    }
    return ret;
  }

  plaidml_value* as_ptr() const { return ptr_.get(); }

 private:
  std::shared_ptr<plaidml_value> ptr_;
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
