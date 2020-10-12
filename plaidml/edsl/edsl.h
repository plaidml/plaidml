// Copyright 2019 Intel Corporation.

#pragma once

#include <algorithm>
#include <memory>
#include <ostream>
#include <string>
#include <utility>
#include <vector>

#include "plaidml/core/core.h"
#include "plaidml/edsl/ffi.h"

namespace plaidml {
namespace edsl {

struct Constraint;
class Contraction;
class IndexedTensor;
class Tensor;
class TensorDim;
class TensorIndex;
struct TensorRef;
class Value;

namespace details {

struct Deleter {
  void operator()(plaidml_dim_expr* ptr) { ffi::call_void(plaidml_dim_expr_free, ptr); }
  void operator()(plaidml_expr* ptr) { ffi::call_void(plaidml_expr_free, ptr); }
  void operator()(plaidml_integers* ptr) { ffi::call_void(plaidml_integers_free, ptr); }
  void operator()(plaidml_poly_expr* ptr) { ffi::call_void(plaidml_poly_expr_free, ptr); }
  void operator()(plaidml_strings* ptr) { ffi::call_void(plaidml_strings_free, ptr); }
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
/// Initializes the PlaidML EDSL API.
///
inline void init() {
  plaidml::init();
  ffi::call_void(plaidml_edsl_init);
}

///
/// Lists the available targets.
///
inline std::vector<std::string> list_targets() {
  auto strs = details::make_ptr(ffi::call<plaidml_strings*>(plaidml_targets_get));
  std::vector<std::string> ret(strs->size);
  for (size_t i = 0; i < ret.size(); i++) {
    ret[i] = ffi::str(strs->elts[i]);
  }
  return ret;
}

///
/// \defgroup edsl_objects Objects
///

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

  TensorDim(plaidml_int_op op, const std::vector<TensorDim>& args) : ptr_(details::make_ptr(makeOp(op, args))) {}

  ///
  /// Represents a subtraction operator overload.
  ///
  TensorDim operator-() const;

  ///
  /// Returns the TensorDim as a string.
  ///
  std::string str() const { return ffi::str(ffi::call<plaidml_string*>(plaidml_dim_expr_repr, as_ptr())); }

  plaidml_dim_expr* as_ptr() const { return ptr_.get(); }

 private:
  static plaidml_dim_expr* makeOp(plaidml_int_op op, const std::vector<TensorDim>& args) {
    std::vector<plaidml_dim_expr*> operands;
    for (const auto& arg : args) {
      operands.push_back(arg.as_ptr());
    }
    return ffi::call<plaidml_dim_expr*>(plaidml_dim_expr_op, op, operands.size(), operands.data());
  }

 private:
  std::shared_ptr<plaidml_dim_expr> ptr_;
};

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
  explicit TensorIndex(int64_t value)
      : ptr_(details::make_ptr(ffi::call<plaidml_poly_expr*>(plaidml_poly_expr_literal, value))) {}

  ///
  /// TensorIndex constructor
  ///
  explicit TensorIndex(const std::string& name = "")
      : ptr_(details::make_ptr(ffi::call<plaidml_poly_expr*>(plaidml_poly_expr_index, name.c_str()))) {}

  TensorIndex(plaidml_int_op op, const std::vector<TensorIndex>& args)
      : ptr_(details::make_ptr(makePolyOp(op, args))) {}

  TensorIndex(plaidml_int_op op, const TensorIndex& idx, const TensorDim& dim, bool lhs_first)
      : ptr_(details::make_ptr(makeDimPolyOp(op, idx, dim, lhs_first))) {}

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
  static plaidml_poly_expr* makePolyOp(plaidml_int_op op, const std::vector<TensorIndex>& args) {
    std::vector<plaidml_poly_expr*> operands;
    for (const auto& arg : args) {
      operands.push_back(arg.as_ptr());
    }
    return ffi::call<plaidml_poly_expr*>(plaidml_poly_expr_op, op, operands.size(), operands.data());
  }

  static plaidml_poly_expr* makeDimPolyOp(plaidml_int_op op, const TensorIndex& idx, const TensorDim& dim,
                                          bool lhs_first) {
    std::vector<plaidml_poly_expr*> operands;
    auto* dim_ptr = ffi::call<plaidml_poly_expr*>(plaidml_poly_expr_dim, dim.as_ptr());
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
/// \class Tensor
/// A multidimensional array of a fixed shape.
///
class Tensor {
 public:
  ///
  /// Tensor constructor
  ///
  Tensor() = default;

  explicit Tensor(plaidml_expr* ptr) : ptr_(details::make_ptr(ptr)) {}

  ///
  /// Tensor constructor
  /// \param value int
  /// \return Tensor
  ///
  explicit Tensor(int value) : ptr_(details::make_ptr(ffi::call<plaidml_expr*>(plaidml_expr_int, value))) {}

  ///
  /// Tensor constructor
  /// \param value unsigned int
  /// \return Tensor
  ///
  explicit Tensor(unsigned value) : ptr_(details::make_ptr(ffi::call<plaidml_expr*>(plaidml_expr_int, value))) {}

  ///
  /// Tensor constructor
  /// \param value uint64_t
  /// \return Tensor
  ///
  explicit Tensor(uint64_t value) : ptr_(details::make_ptr(ffi::call<plaidml_expr*>(plaidml_expr_uint, value))) {}

  ///
  /// Tensor constructor
  /// \param value int64_t
  /// \return Tensor
  ///
  explicit Tensor(int64_t value) : ptr_(details::make_ptr(ffi::call<plaidml_expr*>(plaidml_expr_int, value))) {}

  ///
  /// Tensor constructor
  /// \param value double
  /// \return Tensor
  ///
  explicit Tensor(double value) : ptr_(details::make_ptr(ffi::call<plaidml_expr*>(plaidml_expr_float, value))) {}

  ///
  /// Tensor constructor
  ///
  explicit Tensor(const TensorDim& dim)
      : ptr_(details::make_ptr(ffi::call<plaidml_expr*>(plaidml_expr_dim, dim.as_ptr()))) {}

  template <typename... Ts>
  IndexedTensor operator()(Ts... idxs) const;
  IndexedTensor operator()(const std::vector<TensorIndex>& idxs) const;

  ///
  /// Represents an eltwise negation
  ///
  Tensor operator-() const;

  ///
  /// Represents an eltwise bit_not
  ///
  Tensor operator~() const;

  ///
  /// Elementwise logical not
  ///
  Tensor operator!() const;

  ///
  /// TODO
  ///
  std::string str() const {  //
    return ffi::str(ffi::call<plaidml_string*>(plaidml_expr_repr, as_ptr()));
  }

  ///
  /// Return the tensor's shape
  ///
  TensorShape compute_shape() const { return TensorShape(ffi::call<plaidml_shape*>(plaidml_expr_get_shape, as_ptr())); }

  DType dtype() const { return static_cast<DType>(ffi::call<plaidml_datatype>(plaidml_expr_get_dtype, as_ptr())); }

  size_t rank() const { return ffi::call<size_t>(plaidml_expr_get_rank, as_ptr()); }

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

  ///
  /// Get an element of an operation that returns a tuple (i.e. multlpe results).
  ///
  Tensor element(size_t ordinal) const;

  plaidml_expr* as_ptr() const { return ptr_.get(); }

  void* raw_ptr() const { return ffi::call<void*>(plaidml_expr_ptr, as_ptr()); }

 private:
  std::shared_ptr<plaidml_expr> ptr_;
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
/// \class IndexedTensor
/// This is an IndexedTensor
///
class IndexedTensor {
  friend class Tensor;
  friend class Contraction;

 public:
  IndexedTensor() : op_(PLAIDML_COMBO_OP_NONE) {}

  IndexedTensor(const Tensor& ref, const std::vector<TensorIndex>& idxs)
      : op_(PLAIDML_COMBO_OP_NONE), ref_(ref), idxs_(idxs) {}

  IndexedTensor(plaidml_combo_op op, const std::vector<IndexedTensor>& operands) : op_(op), operands_(operands) {}

  ///
  /// Performs an addition combination within a contraction.
  ///
  IndexedTensor operator+(const IndexedTensor& rhs) const { return IndexedTensor(PLAIDML_COMBO_OP_ADD, {*this, rhs}); }

  ///
  /// Performs a multiplication combination within a contraction.
  ///
  IndexedTensor operator*(const IndexedTensor& rhs) const { return IndexedTensor(PLAIDML_COMBO_OP_MUL, {*this, rhs}); }

  ///
  /// Performs an equality comparison combination within a contraction.
  ///
  IndexedTensor operator==(const IndexedTensor& rhs) const { return IndexedTensor(PLAIDML_COMBO_OP_EQ, {*this, rhs}); }

 private:
  plaidml_combo_op op_;
  std::vector<IndexedTensor> operands_;
  Tensor ref_;
  std::vector<TensorIndex> idxs_;
};

// Performs a conditional combination within a contraction.
inline IndexedTensor cond(const IndexedTensor& lhs, const IndexedTensor& rhs, const IndexedTensor& true_case) {
  return IndexedTensor(PLAIDML_COMBO_OP_COND, {lhs, rhs, true_case});
}

class Contraction {
 public:
  explicit Contraction(const std::string& name = "") : name_(name) {}

  Contraction(const std::vector<TensorDim>& dims, const std::vector<TensorIndex>& idxs, const std::string& name = "")
      : name_(name) {
    outShape(dims);
    outAccess(idxs);
  }

  template <typename... Ts>
  Contraction& outShape(Ts... idxs);
  Contraction& outShape(const std::vector<TensorDim>& dims);
  Contraction& outShape(const std::vector<int64_t>& dims);

  template <typename... Ts>
  Contraction& outAccess(Ts... idxs);
  Contraction& outAccess(const std::vector<TensorIndex>& idxs);

  ///
  /// Performs an assignment contraction.
  ///
  Contraction& assign(const IndexedTensor& tensor);

  ///
  /// Performs an maximize contraction.
  ///
  Contraction& max(const IndexedTensor& tensor);

  ///
  /// Performs an minimize contraction.
  ///
  Contraction& min(const IndexedTensor& tensor);

  ///
  /// Performs a product contraction.
  ///
  Contraction& product(const IndexedTensor& tensor);

  ///
  /// Performs a summation contraction.
  ///
  Contraction& sum(const IndexedTensor& tensor);

  ///
  /// Set the initializer for a contraction.
  ///
  Contraction& init(const Tensor& rhs);

  ///
  /// Add a constraint to a contraction.
  ///
  Contraction& add_constraint(const Constraint& constraint);

  ///
  /// Add constraints to a contraction.
  ///
  Contraction& add_constraints(const std::vector<Constraint>& constraints);

  ///
  /// Construct a contraction.
  ///
  Tensor build();

  operator Tensor() { return build(); }

 private:
  std::string name_;
  std::vector<TensorDim> outDims_;
  std::vector<TensorIndex> outIdxs_;
  std::vector<Constraint> constraints_;
  IndexedTensor rhs_;
  plaidml_agg_op agg_op_;
  Tensor init_;
};

template <typename... Ts>
inline Contraction& Contraction::outShape(Ts... dims) {
  details::into_vector(&outDims_, std::forward<Ts>(dims)...);
  return *this;
}

inline Contraction& Contraction::outShape(const std::vector<TensorDim>& dims) {
  outDims_ = dims;
  return *this;
}

inline Contraction& Contraction::outShape(const std::vector<int64_t>& dims) {
  for (int64_t dim : dims) {
    outDims_.emplace_back(dim);
  }
  return *this;
}

template <typename... Ts>
inline Contraction& Contraction::outAccess(Ts... idxs) {
  details::into_vector(&outIdxs_, std::forward<Ts>(idxs)...);
  return *this;
}

inline Contraction& Contraction::outAccess(const std::vector<TensorIndex>& idxs) {
  outIdxs_ = idxs;
  return *this;
}

inline Contraction& Contraction::init(const Tensor& rhs) {
  init_ = rhs;
  return *this;
}

inline Contraction& Contraction::assign(const IndexedTensor& rhs) {
  agg_op_ = PLAIDML_AGG_OP_ASSIGN;
  rhs_ = rhs;
  return *this;
}

inline Contraction& Contraction::max(const IndexedTensor& rhs) {
  agg_op_ = PLAIDML_AGG_OP_MAX;
  rhs_ = rhs;
  return *this;
}

inline Contraction& Contraction::min(const IndexedTensor& rhs) {
  agg_op_ = PLAIDML_AGG_OP_MIN;
  rhs_ = rhs;
  return *this;
}

inline Contraction& Contraction::product(const IndexedTensor& rhs) {
  agg_op_ = PLAIDML_AGG_OP_PROD;
  rhs_ = rhs;
  return *this;
}

inline Contraction& Contraction::sum(const IndexedTensor& rhs) {
  agg_op_ = PLAIDML_AGG_OP_SUM;
  rhs_ = rhs;
  return *this;
}

inline Contraction& Contraction::add_constraint(const Constraint& constraint) {
  constraints_.push_back(constraint);
  return *this;
}

inline Contraction& Contraction::add_constraints(const std::vector<Constraint>& constraints) {
  for (const auto& constraint : constraints) {
    add_constraint(constraint);
  }
  return *this;
}

inline Tensor Contraction::build() {
  size_t rank = outDims_.size();
  if (rank != outIdxs_.size()) {
    throw std::runtime_error("Rank mismatch between outShape and outAccess");
  }
  std::vector<plaidml_poly_expr*> idxs(rank);
  std::vector<plaidml_dim_expr*> dims(rank);
  for (size_t i = 0; i < rank; i++) {
    idxs[i] = outIdxs_[i].as_ptr();
    dims[i] = outDims_[i].as_ptr();
  }

  auto* ptr = ffi::call<plaidml_expr*>(  //
      plaidml_expr_contraction,          //
      agg_op_,                           //
      rhs_.op_,                          //
      rank,                              //
      idxs.data(),                       //
      dims.data(),                       //
      init_.as_ptr(),                    //
      name_.c_str());

  std::vector<IndexedTensor> operands;
  if (rhs_.op_ == PLAIDML_COMBO_OP_NONE) {
    operands.push_back(rhs_);
  } else {
    operands = rhs_.operands_;
  }

  for (const IndexedTensor& operand : operands) {
    size_t rank = operand.idxs_.size();
    std::vector<plaidml_poly_expr*> idxs(rank);
    for (size_t i = 0; i < rank; i++) {
      idxs[i] = operand.idxs_[i].as_ptr();
    }
    ffi::call_void(                       //
        plaidml_contraction_add_operand,  //
        ptr,                              //
        operand.ref_.as_ptr(),            //
        idxs.size(),                      //
        idxs.data());
  }

  for (const Constraint& constraint : constraints_) {
    ffi::call_void(                          //
        plaidml_contraction_add_constraint,  //
        ptr,                                 //
        constraint.lhs.as_ptr(),             //
        constraint.rhs.as_ptr());
  }

  ffi::call_void(plaidml_contraction_build, ptr);
  return Tensor(ptr);
}

template <typename... Ts>
inline IndexedTensor Tensor::operator()(Ts... idxs) const {
  std::vector<TensorIndex> vec;
  details::into_vector(&vec, std::forward<Ts>(idxs)...);
  return IndexedTensor(*this, vec);
}
inline IndexedTensor Tensor::operator()(const std::vector<TensorIndex>& idxs) const {
  return IndexedTensor(*this, idxs);
}

inline Tensor Tensor::element(size_t ordinal) const {
  return Tensor(ffi::call<plaidml_expr*>(plaidml_expr_element, as_ptr(), ordinal));
}

inline Tensor Constant(    //
    const Buffer& buffer,  //
    const std::string& name) {
  auto* ptr = ffi::call<plaidml_expr*>(  //
      plaidml_expr_constant,             //
      buffer.as_ptr(),                   //
      name.c_str());
  return Tensor(ptr);
}

inline Tensor Constant(int value) { return Tensor(value); }

inline Tensor Constant(int64_t value) { return Tensor(value); }

inline Tensor Constant(uint64_t value) { return Tensor(value); }

inline Tensor Constant(double value) { return Tensor(value); }

inline Tensor Placeholder(     //
    const TensorShape& shape,  //
    const std::string& name = "") {
  auto* ptr = ffi::call<plaidml_expr*>(  //
      plaidml_expr_input,                //
      shape.as_ptr(),                    //
      name.c_str());
  return Tensor(ptr);
}

inline Tensor Placeholder(             //
    DType dtype,                       //
    const std::vector<int64_t>& dims,  //
    const std::string& name = "") {
  TensorShape shape(dtype, dims);
  return Placeholder(shape, name);
}

inline Tensor Zero() { return Tensor(0); }

Tensor intrinsicCall(const std::string& fn, const std::vector<Tensor>& args);

template <typename... Ts>
Tensor intrinsic(const std::string& fn, Ts... args) {
  std::vector<Tensor> vec;
  details::into_vector(&vec, std::forward<Ts>(args)...);
  return intrinsicCall(fn, vec);
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
inline Tensor abs(const Tensor& x) { return intrinsic("abs", x); }

///
/// Computes the elementwise arccosine of `x`.
/// \param x Tensor
/// \return Tensor
///
inline Tensor acos(const Tensor& x) { return intrinsic("acos", x); }

///
/// Computes the elementwise arcsine of `x`.
/// \param x Tensor
/// \return Tensor
///
inline Tensor asin(const Tensor& x) { return intrinsic("asin", x); }

///
/// Computes the elementwise arctangent of `x`.
/// \param x Tensor
/// \return Tensor
///
inline Tensor atan(const Tensor& x) { return intrinsic("atan", x); }

///
/// Casts the element type of a tensor `x` to the type specified by `dtype`.
/// \param x Tensor
/// \param dtype DType
/// \return Tensor
///
inline Tensor cast(const Tensor& x, DType dtype) {
  return Tensor{ffi::call<plaidml_expr*>(plaidml_expr_cast, x.as_ptr(), static_cast<plaidml_datatype>(dtype))};
}

///
/// Computes the elementwise ceiling of `x`.
/// \param x Tensor
/// \return Tensor
///
inline Tensor ceil(const Tensor& x) { return intrinsic("ceil", x); }

///
/// Computes the elementwise cosine of `x`.
/// \param x Tensor
/// \return Tensor
///
inline Tensor cos(const Tensor& x) { return intrinsic("cos", x); }

///
/// Computes the elementwise hyperbolic cosine of `x`.
/// \param x Tensor
/// \return Tensor
///
inline Tensor cosh(const Tensor& x) { return intrinsic("cosh", x); }

///
/// Computes the elementwise Gauss error function of `x`
/// \param x Tensor
/// \return Tensor
///
inline Tensor erf(const Tensor& x) { return intrinsic("erf", x); }

///
/// Computes the elementwise natural exponential function of `x`: _e_<sup>x</sup>.
/// \param x Tensor
/// \return Tensor
///
inline Tensor exp(const Tensor& x) { return intrinsic("exp", x); }

///
/// Computes the elementwise floor of `x`.
/// \param x Tensor
/// \return Tensor
///
inline Tensor floor(const Tensor& x) { return intrinsic("floor", x); }

///
/// Takes an input tensor (`x`) and a set of indices to gather over (`y`), and returns an output tensor that gathers the
/// input tensor from the indices specified.
/// \param x Tensor
/// \param y Tensor
/// \return Tensor
///
inline Tensor gather(const Tensor& x, const Tensor& y) { return intrinsic("gather", x, y); }

///
/// Returns the identity of `x`.
/// \param x Tensor
/// \return Tensor
///
inline Tensor ident(const Tensor& x) { return intrinsic("ident", x); }

///
/// Returns a tensor populated with the index value of the shape and axis specified.
/// \param dims std::vector<TensorDim>
/// \param axis size_t
/// \return Tensor
///
inline Tensor index(const std::vector<TensorDim>& dims, size_t axis) {
  std::vector<Tensor> args = {Tensor{static_cast<int64_t>(axis)}};
  for (const auto& dim : dims) {
    args.emplace_back(dim);
  }
  return intrinsicCall("index", args);
}

///
/// Computes the elementwise natural logarithm of `x`: _ln_(`x`).
/// \param x Tensor
/// \return Tensor
///
inline Tensor log(const Tensor& x) { return intrinsic("log", x); }

///
/// Computes the elementwise `y`th power of `x`.
/// \param x Tensor
/// \param y Tensor
/// \return Tensor
///
inline Tensor pow(const Tensor& x, const Tensor& y) { return intrinsic("pow", x, y); }

///
/// Generates a Tensor of elementwise pseudorandom numbers using the seed values specified in `state`.
/// \param state Tensor
/// \param dims vector<int64_t>
/// \return Tensor
///
inline std::pair<Tensor, Tensor> prng(const Tensor& state, const std::vector<int64_t>& dims) {
  std::vector<Tensor> args = {state};
  for (int64_t dim : dims) {
    args.emplace_back(TensorDim(dim));
  }
  Tensor R = intrinsicCall("prng", args);
  return std::make_pair(R.element(0), R.element(1));
}

///
/// Takes an input tensor `x` and reshapes it according to `dims`.
/// \param x Tensor
/// \param dims vector<int64_t>
/// \return Tensor
///
inline Tensor reshape(const Tensor& x, const std::vector<int64_t>& dims) {
  std::vector<Tensor> args = {x};
  for (int64_t dim : dims) {
    args.emplace_back(dim);
  }
  return intrinsicCall("reshape", args);
}

///
/// Takes an input tensor `x` and reshapes it according to `dims`.
/// \param x Tensor
/// \param dims vector<TensorDim>
/// \return Tensor
///
inline Tensor reshape(const Tensor& x, const std::vector<TensorDim>& dims) {
  std::vector<Tensor> args = {x};
  for (const TensorDim& dim : dims) {
    args.emplace_back(dim);
  }
  return intrinsicCall("reshape", args);
}

///
/// Rounds `x` elementwise.
/// \param x Tensor
/// \return Tensor
///
inline Tensor round(const Tensor& x) { return intrinsic("round", x); }

///
/// Takes an input tensor (`x`), a set of indices to scatter over (`y`), and the number of elements in the scattered
/// tensor (`z`), and returns an output tensor that scatters the input tensor across the number of elements specified.
/// \param x Tensor
/// \param y Tensor
/// \param z Tensor
/// \return Tensor
///
inline Tensor scatter(const Tensor& x, const Tensor& y, const Tensor& z) { return intrinsic("scatter", x, y, z); }

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
  return intrinsic("select", cond, true_case, false_case);
}

///
/// Returns the shape of `x` as a Tensor.
/// \param x Tensor
/// \return Tensor
///
inline Tensor shape(const Tensor& x) { return intrinsic("shape", x); }

///
/// Computes the elementwise sine of `x`.
/// \param x Tensor
/// \return Tensor
///
inline Tensor sin(const Tensor& x) { return intrinsic("sin", x); }

///
/// Computes the elementwise hyperbolic sine of `x`.
/// \param x Tensor
/// \return Tensor
///
inline Tensor sinh(const Tensor& x) { return intrinsic("sinh", x); }

///
/// Computes the elementwise square root of `x`.
/// \param x Tensor
/// \return Tensor
///
inline Tensor sqrt(const Tensor& x) { return intrinsic("sqrt", x); }

///
/// Computes the elementwise tangent of `x`.
/// \param x Tensor
/// \return Tensor
///
inline Tensor tan(const Tensor& x) { return intrinsic("tan", x); }

///
/// Computes the elementwise hyperbolic tangent of `x`.
/// \param x Tensor
/// \return Tensor
///
inline Tensor tanh(const Tensor& x) { return intrinsic("tanh", x); }

///
/// Adds a tracepoint to the graph
///
inline Tensor trace(const Tensor& x, const std::string& msg) {
  return Tensor{ffi::call<plaidml_expr*>(plaidml_expr_trace, x.as_ptr(), msg.c_str())};
}

/// @}

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
    return intrinsic(_fn_, lhs, Tensor(rhs));                                        \
  }                                                                                  \
  inline Tensor operator _op_(const TensorDim& lhs, const Tensor& rhs) { /**/        \
    return intrinsic(_fn_, Tensor(lhs), rhs);                                        \
  }                                                                                  \
  inline TensorDim operator _op_(const TensorDim& lhs, const TensorDim& rhs) {       \
    return TensorDim(_int_op_, {lhs, rhs});                                          \
  }                                                                                  \
  inline TensorDim operator _op_(int64_t lhs, const TensorDim& rhs) {                \
    return TensorDim(_int_op_, {TensorDim(lhs), rhs});                               \
  }                                                                                  \
  inline TensorDim operator _op_(const TensorDim& lhs, int64_t rhs) {                \
    return TensorDim(_int_op_, {lhs, TensorDim(rhs)});                               \
  }

PLAIDML_EDSL_DEFINE_TENSOR_IDXDIM_BINARY_OPS(+, PLAIDML_INT_OP_ADD, "add");
PLAIDML_EDSL_DEFINE_TENSOR_IDXDIM_BINARY_OPS(-, PLAIDML_INT_OP_SUB, "sub");
PLAIDML_EDSL_DEFINE_TENSOR_IDXDIM_BINARY_OPS(*, PLAIDML_INT_OP_MUL, "mul");
PLAIDML_EDSL_DEFINE_TENSOR_IDXDIM_BINARY_OPS(/, PLAIDML_INT_OP_DIV, "div");

#define PLAIDML_EDSL_DEFINE_TENSOR_DIM_BINARY_FN(_fn_, _int_op_)                                                  \
  inline TensorDim _fn_(const TensorDim& lhs, const TensorDim& rhs) { return TensorDim(_int_op_, {lhs, rhs}); }   \
  inline TensorDim _fn_(int64_t lhs, const TensorDim& rhs) { return TensorDim(_int_op_, {TensorDim(lhs), rhs}); } \
  inline TensorDim _fn_(const TensorDim& lhs, int64_t rhs) { return TensorDim(_int_op_, {lhs, TensorDim(rhs)}); }

PLAIDML_EDSL_DEFINE_TENSOR_DIM_BINARY_FN(max, PLAIDML_INT_OP_MAX);
PLAIDML_EDSL_DEFINE_TENSOR_DIM_BINARY_FN(min, PLAIDML_INT_OP_MIN);

inline Tensor Tensor::operator-() const { return intrinsicCall("neg", {*this}); }
inline Tensor Tensor::operator~() const { return intrinsicCall("bit_not", {*this}); }
inline Tensor Tensor::operator!() const { return intrinsicCall("logical_not", {*this}); }

#define PLAIDML_EDSL_DEFINE_TENSOR_BINARY_OPS(_op_, _fn_)                                                 \
  inline Tensor operator _op_(const Tensor& lhs, const Tensor& rhs) { return intrinsic(_fn_, lhs, rhs); } \
  inline Tensor operator _op_(const Tensor& lhs, int rhs) { return intrinsic(_fn_, lhs, rhs); }           \
  inline Tensor operator _op_(const Tensor& lhs, int64_t rhs) { return intrinsic(_fn_, lhs, rhs); }       \
  inline Tensor operator _op_(const Tensor& lhs, uint64_t rhs) { return intrinsic(_fn_, lhs, rhs); }      \
  inline Tensor operator _op_(const Tensor& lhs, double rhs) { return intrinsic(_fn_, lhs, rhs); }        \
  inline Tensor operator _op_(int lhs, const Tensor& rhs) { return intrinsic(_fn_, lhs, rhs); }           \
  inline Tensor operator _op_(int64_t lhs, const Tensor& rhs) { return intrinsic(_fn_, lhs, rhs); }       \
  inline Tensor operator _op_(uint64_t lhs, const Tensor& rhs) { return intrinsic(_fn_, lhs, rhs); }      \
  inline Tensor operator _op_(double lhs, const Tensor& rhs) { return intrinsic(_fn_, lhs, rhs); }

PLAIDML_EDSL_DEFINE_TENSOR_BINARY_OPS(+, "add");
PLAIDML_EDSL_DEFINE_TENSOR_BINARY_OPS(-, "sub");
PLAIDML_EDSL_DEFINE_TENSOR_BINARY_OPS(*, "mul");
PLAIDML_EDSL_DEFINE_TENSOR_BINARY_OPS(/, "div");
PLAIDML_EDSL_DEFINE_TENSOR_BINARY_OPS(%, "mod");
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
PLAIDML_EDSL_DEFINE_TENSOR_BINARY_OPS(&&, "logical_and");
PLAIDML_EDSL_DEFINE_TENSOR_BINARY_OPS(||, "logical_or");

inline Tensor intrinsicCall(const std::string& fn, const std::vector<Tensor>& args) {
  std::vector<plaidml_expr*> ptrs(args.size());
  for (size_t i = 0; i < args.size(); i++) {
    ptrs[i] = args[i].as_ptr();
  }
  return Tensor{ffi::call<plaidml_expr*>(plaidml_expr_intrinsic, fn.c_str(), ptrs.size(), ptrs.data())};
}

class Value {
 public:
  Value() : ptr_(details::make_ptr(ffi::call<plaidml_value*>(plaidml_value_none))) {}

  explicit Value(plaidml_value* ptr) : ptr_(details::make_ptr(ffi::call<plaidml_value*>(plaidml_value_clone, ptr))) {}

  explicit Value(int value) : ptr_(details::make_ptr(ffi::call<plaidml_value*>(plaidml_value_int, value))) {}

  explicit Value(size_t value) : ptr_(details::make_ptr(ffi::call<plaidml_value*>(plaidml_value_int, value))) {}

  explicit Value(int64_t value) : ptr_(details::make_ptr(ffi::call<plaidml_value*>(plaidml_value_int, value))) {}

  explicit Value(double value) : ptr_(details::make_ptr(ffi::call<plaidml_value*>(plaidml_value_float, value))) {}

  explicit Value(const std::string& value)
      : ptr_(details::make_ptr(ffi::call<plaidml_value*>(plaidml_value_str, value.c_str()))) {}

  explicit Value(const TensorDim& dim)
      : ptr_(details::make_ptr(ffi::call<plaidml_value*>(plaidml_value_dim, dim.as_ptr()))) {}

  explicit Value(const Tensor& tensor) {
    if (auto* ptr = tensor.as_ptr()) {
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

  double as_float() const {  //
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
    std::vector<Value> ret(tuple->size);
    for (size_t i = 0; i < ret.size(); i++) {
      ret[i] = Value{tuple->elts[i]};
    }
    return ret;
  }

  std::vector<int64_t> as_int_tuple() const {
    auto tuple = details::make_ptr(ffi::call<plaidml_tuple*>(plaidml_value_tuple_get, as_ptr()));
    std::vector<int64_t> ret(tuple->size);
    for (size_t i = 0; i < ret.size(); i++) {
      ret[i] = Value{tuple->elts[i]}.as_int();
    }
    return ret;
  }

  std::vector<int64_t> as_int_tuple_or_empty() const {  //
    return is_none() ? std::vector<int64_t>{} : as_int_tuple();
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

inline Program buildProgram(const std::string& name, const std::vector<Tensor>& inputs,
                            const std::vector<Tensor>& outputs) {
  std::vector<plaidml_expr*> input_exprs(inputs.size());
  for (size_t i = 0; i < inputs.size(); i++) {
    input_exprs[i] = inputs[i].as_ptr();
  }
  std::vector<plaidml_expr*> output_exprs(outputs.size());
  for (size_t i = 0; i < outputs.size(); i++) {
    output_exprs[i] = outputs[i].as_ptr();
  }
  auto* ptr = ffi::call<plaidml_program*>(  //
      plaidml_build,                        //
      name.c_str(),                         //
      inputs.size(),                        //
      input_exprs.data(),                   //
      /*shapes=*/nullptr,                   //
      outputs.size(),                       //
      output_exprs.data());
  return Program(ptr);
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

inline std::ostream& operator<<(std::ostream& os, const Value& x) {
  os << x.str();
  return os;
}

}  // namespace edsl
}  // namespace plaidml
