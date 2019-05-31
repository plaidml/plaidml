// Copyright 2019 Intel Corporation.

#pragma once

#include <algorithm>
#include <functional>
#include <iterator>
#include <memory>
#include <ostream>
#include <sstream>
#include <string>
#include <utility>
#include <vector>

#include "plaidml/edsl/ffi.h"

namespace vertexai {
namespace plaidml {
namespace edsl {

class IndexedTensor;
class Tensor;
class TensorDim;
class TensorFriend;
class TensorIndex;

namespace ffi {

inline std::string str(tile_string* ptr) {
  std::string ret{tile_string_ptr(ptr)};
  tile_string_free(ptr);
  return ret;
}

template <typename T, typename F, typename... Args>
T call(F fn, Args... args) {
  tile_error err;
  auto ret = fn(&err, args...);
  if (err.code) {
    throw std::runtime_error(str(err.msg));
  }
  return ret;
}

template <typename F, typename... Args>
void call_void(F fn, Args... args) {
  tile_error err;
  fn(&err, args...);
  if (err.code) {
    throw std::runtime_error(str(err.msg));
  }
}

}  // namespace ffi

namespace details {

template <typename T>
struct Deleter {
  std::function<void(tile_error*, T*)> fn;
  void operator()(T* ptr) { ffi::call_void(fn, ptr); }
};

inline std::shared_ptr<tile_shape> make_tile_shape(tile_shape* ptr) {
  return std::shared_ptr<tile_shape>(ptr, Deleter<tile_shape>{tile_shape_free});
}

inline std::shared_ptr<tile_expr> make_tile_expr(tile_expr* ptr) {
  return std::shared_ptr<tile_expr>(ptr, Deleter<tile_expr>{tile_expr_free});
}

inline std::shared_ptr<tile_poly_expr> make_tile_poly_expr(tile_poly_expr* ptr) {
  return std::shared_ptr<tile_poly_expr>(ptr, Deleter<tile_poly_expr>{tile_poly_expr_free});
}

inline std::shared_ptr<tile_program> make_tile_program(tile_program* ptr) {
  return std::shared_ptr<tile_program>(ptr, Deleter<tile_program>{tile_program_free});
}

template <typename T>
void into_vector(std::vector<T>*) {}

template <typename T, typename Head, typename... Tail>
void into_vector(std::vector<T>* into, Head&& head, Tail&&... tail) {
  into->emplace_back(std::forward<Head>(head));
  into_vector(into, std::forward<Tail>(tail)...);
}

}  // namespace details

class TensorShape {
  friend class Tensor;
  friend class TensorFriend;

 public:
  TensorShape(plaidml_datatype dtype,              //
              const std::vector<uint64_t>& sizes,  //
              const std::string& layout = "")
      : ptr_(details::make_tile_shape(ffi::call<tile_shape*>(tile_shape_alloc, dtype, layout.c_str()))) {
    size_t stride = 1;
    std::vector<int64_t> strides(sizes.size());
    for (int i = sizes.size() - 1; i >= 0; --i) {
      strides[i] = stride;
      stride *= sizes[i];
    }
    for (size_t i = 0; i < sizes.size(); i++) {
      ffi::call_void(tile_shape_add_dimension, ptr_.get(), sizes[i], strides[i]);
    }
  }

  TensorShape(plaidml_datatype dtype,               //
              const std::vector<uint64_t>& sizes,   //
              const std::vector<int64_t>& strides,  //
              const std::string& layout = "")
      : ptr_(details::make_tile_shape(ffi::call<tile_shape*>(tile_shape_alloc, dtype, layout.c_str()))) {
    if (sizes.size() != strides.size()) {
      throw std::runtime_error("Sizes and strides must have same rank.");
    }
    for (size_t i = 0; i < sizes.size(); i++) {
      ffi::call_void(tile_shape_add_dimension, ptr_.get(), sizes[i], strides[i]);
    }
  }

  plaidml_datatype type() const { return ffi::call<plaidml_datatype>(tile_shape_get_type, ptr_.get()); }
  size_t rank() const { return ffi::call<size_t>(tile_shape_get_rank, ptr_.get()); }
  uint64_t size_at(size_t dim) const { return ffi::call<uint64_t>(tile_shape_get_dimension_size, ptr_.get(), dim); }
  int64_t stride_at(size_t dim) const { return ffi::call<int64_t>(tile_shape_get_dimension_stride, ptr_.get(), dim); }
  uint64_t byte_size() const { return ffi::call<uint64_t>(tile_shape_get_byte_size, ptr_.get()); }
  std::string str() const { return ffi::str(ffi::call<tile_string*>(tile_shape_repr, ptr_.get())); }
  const void* ptr() const { return ffi::call<const void*>(tile_shape_get_ptr, ptr_.get()); }
  bool operator==(const TensorShape& rhs) const { return str() == rhs.str(); }

 private:
  explicit TensorShape(const std::shared_ptr<tile_shape>& ptr) : ptr_(ptr) {}

 private:
  std::shared_ptr<tile_shape> ptr_;
};

class Program {
 public:
  Program(const std::string& name, const std::vector<Tensor>& tensors);
  std::string str() const { return ffi::str(ffi::call<tile_string*>(tile_program_repr, ptr_.get())); }
  const void* runinfo() const { return ffi::call<const void*>(tile_program_runinfo, ptr_.get()); }

 private:
  std::shared_ptr<tile_program> ptr_;
};

class TensorIndexIterator {
 public:
  TensorIndexIterator() = default;
  explicit TensorIndexIterator(TensorIndex* index) : index_(index) {}

  TensorIndexIterator& operator++() {
    index_ = nullptr;
    return *this;
  }

  bool operator!=(const TensorIndexIterator& other) { return index_ != other.index_; }
  TensorIndex& operator*() { return *index_; }

 private:
  TensorIndex* index_ = nullptr;
};

class Constraint {
  friend class TensorFriend;
  friend class TensorIndex;

 public:
  Constraint operator&&(const Constraint& rhs) const { return Constraint(); }
  operator bool() const { return true; }

 private:
  Constraint() = default;
};

class TensorDim {
  friend class Tensor;
  friend class TensorFriend;
  friend class TensorIndex;

  struct Impl {
    std::unique_ptr<size_t> size;
  };

 public:
  TensorDim() : impl_(new Impl) {}
  explicit TensorDim(size_t value) : impl_(new Impl) { impl_->size.reset(new size_t(value)); }

  TensorDim operator-() const;

 private:
  std::shared_ptr<Impl> impl_;
};

class TensorIndex {
  friend class Tensor;
  friend class TensorFriend;

  struct Impl {
    std::shared_ptr<tile_poly_expr> ptr;
  };

 public:
  TensorIndex() : impl_(new Impl) {
    impl_->ptr = details::make_tile_poly_expr(ffi::call<tile_poly_expr*>(tile_poly_expr_index, ""));
  }

  explicit TensorIndex(size_t value) : impl_(new Impl) {
    impl_->ptr = details::make_tile_poly_expr(ffi::call<tile_poly_expr*>(tile_poly_expr_literal, value));
  }

  explicit TensorIndex(const std::string& name) : impl_(new Impl) {
    impl_->ptr = details::make_tile_poly_expr(ffi::call<tile_poly_expr*>(tile_poly_expr_index, name.c_str()));
  }

  TensorIndexIterator begin() { return TensorIndexIterator(this); }
  TensorIndexIterator end() { return TensorIndexIterator{}; }

  TensorIndex operator-() const;

  Constraint operator<(size_t rhs) const {
    ffi::call_void(tile_poly_expr_add_constraint, impl_->ptr.get(), rhs);
    return Constraint();
  }

  Constraint operator<(const TensorDim& rhs) const {
    if (!rhs.impl_->size) {
      throw std::runtime_error("Undefined dimension.");
    }
    ffi::call_void(tile_poly_expr_add_constraint, impl_->ptr.get(), *rhs.impl_->size);
    return Constraint();
  }

  std::string str() const {  //
    return ffi::str(ffi::call<tile_string*>(tile_poly_expr_repr, impl_->ptr.get()));
  }

 private:
  std::shared_ptr<Impl> impl_;
  explicit TensorIndex(const std::shared_ptr<Impl>& impl) : impl_(impl) {}
};

class IndexedTensor {
  friend class Tensor;
  friend class TensorFriend;

  struct ComboParts {
    tile_combo_op op;
    std::vector<tile_expr*> args;
  };

  struct Impl {
    std::shared_ptr<tile_expr> unary;
    std::shared_ptr<ComboParts> nary;
    const Tensor* src = nullptr;
    void MakeContraction(tile_agg_op agg_op, const IndexedTensor& rhs);
  };

 public:
  ~IndexedTensor() = default;

  // Movable constructor
  IndexedTensor(IndexedTensor&& rhs) noexcept : impl_(std::move(rhs.impl_)) {}

  // Represents an aggregation_op of SUM in a contraction
  IndexedTensor& operator+=(const IndexedTensor& rhs) {
    impl_->MakeContraction(TILE_AGG_OP_SUM, rhs);
    return *this;
  }

  // Represents an aggregation_op of PROD in a contraction
  IndexedTensor& operator*=(const IndexedTensor& rhs) {
    impl_->MakeContraction(TILE_AGG_OP_PROD, rhs);
    return *this;
  }

  // Represents an aggregation_op of MAX in a contraction
  IndexedTensor& operator>=(const IndexedTensor& rhs) {
    impl_->MakeContraction(TILE_AGG_OP_MAX, rhs);
    return *this;
  }

  // Represents an aggregation_op of MIN in a contraction
  IndexedTensor& operator<=(const IndexedTensor& rhs) {
    impl_->MakeContraction(TILE_AGG_OP_MIN, rhs);
    return *this;
  }

  // Represents an aggregation_op of ASSIGN in a contraction
  IndexedTensor& operator=(const IndexedTensor& rhs) {
    impl_->MakeContraction(TILE_AGG_OP_ASSIGN, rhs);
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

class Tensor {
  friend class IndexedTensor;
  friend class TensorFriend;

  struct Impl {
    std::shared_ptr<tile_expr> ptr;
    std::vector<TensorDim> dims;
    std::string name;
  };

 public:
  ~Tensor() = default;

  explicit Tensor(int value) : impl_(new Impl) {  //
    impl_->ptr = details::make_tile_expr(ffi::call<tile_expr*>(tile_expr_int, value));
  }

  explicit Tensor(int64_t value) : impl_(new Impl) {
    impl_->ptr = details::make_tile_expr(ffi::call<tile_expr*>(tile_expr_int, value));
  }

  explicit Tensor(double value) : impl_(new Impl) {
    impl_->ptr = details::make_tile_expr(ffi::call<tile_expr*>(tile_expr_float, value));
  }

  explicit Tensor(const TensorShape& shape) : impl_(new Impl) {
    impl_->ptr = details::make_tile_expr(  //
        ffi::call<tile_expr*>(             //
            tile_expr_param,               //
            shape.ptr_.get(),              //
            ""));
  }

  explicit Tensor(const std::vector<TensorDim>& dims) : impl_(new Impl) { impl_->dims = dims; }

  explicit Tensor(const std::initializer_list<TensorDim>& dims) : impl_(new Impl) { impl_->dims = dims; }

  Tensor(const std::string& name, const TensorShape& shape) : impl_(new Impl) {
    impl_->ptr = details::make_tile_expr(  //
        ffi::call<tile_expr*>(             //
            tile_expr_param,               //
            shape.ptr_.get(),              //
            name.c_str()));
  }

  Tensor(const std::string& name, const std::vector<TensorDim>& dims) : impl_(new Impl) {
    impl_->dims = dims;
    impl_->name = name;
  }

  Tensor(const std::string& name, const std::initializer_list<TensorDim>& dims) : impl_(new Impl) {
    impl_->dims = dims;
    impl_->name = name;
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
    std::vector<size_t> sizes;
    for (const auto& dim : impl_->dims) {
      if (!dim.impl_->size) {
        throw std::runtime_error("Undefined dimension.");
      }
      sizes.emplace_back(*dim.impl_->size);
    }
    std::vector<tile_poly_expr*> idx_ptrs(idxs.size());
    for (size_t i = 0; i < idxs.size(); i++) {
      idx_ptrs[i] = idxs[i].impl_->ptr.get();
    }
    auto impl = std::make_unique<IndexedTensor::Impl>();
    impl->src = this;
    impl->unary = details::make_tile_expr(  //
        ffi::call<tile_expr*>(              //
            tile_expr_tensor_spec,          //
            impl_->ptr.get(),               //
            idx_ptrs.size(),                //
            idx_ptrs.data(),                //
            sizes.data()));
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
    return ffi::str(ffi::call<tile_string*>(tile_expr_repr, impl_->ptr.get()));
  }

  // Enable no_defract on a contraction
  Tensor& no_defract() {
    ffi::call_void(tile_expr_contraction_set_no_defract, impl_->ptr.get(), true);
    return *this;
  }

  // Set use_default on a contraction
  Tensor& use_default(const Tensor& rhs) {
    ffi::call_void(tile_expr_contraction_set_use_default, impl_->ptr.get(), rhs.impl_->ptr.get());
    return *this;
  }

  // Return the tensor's shape
  TensorShape shape() const {
    auto ptr = details::make_tile_shape(ffi::call<tile_shape*>(tile_expr_evaluate_shape, impl_->ptr.get()));
    return TensorShape(ptr);
  }

  // Return the size of the tensor's shape at the specified dimension.
  size_t dims(const size_t dim) const {
    auto this_shape = shape();
    if (this_shape.rank() <= dim) {
      throw std::runtime_error("Requested dimension number higher than number of tensor dimensions");
    }
    return this_shape.size_at(dim);
  }

  // Verify that the specified dims match the dims of this tensor.
  void bind_dims(const std::vector<TensorDim>& dims) const {
    std::vector<size_t> sizes;
    if (impl_->dims.size()) {
      // this handles intermediate temporaries (results of previous outputs)
      for (const auto& dim : impl_->dims) {
        if (!dim.impl_->size) {
          throw std::runtime_error("Undefined dimension.");
        }
        sizes.emplace_back(*dim.impl_->size);
      }
    } else {
      // this is the fallback which handles any other case
      auto this_shape = shape();
      for (size_t i = 0; i < this_shape.rank(); i++) {
        sizes.emplace_back(this_shape.size_at(i));
      }
    }

    if (dims.size() != sizes.size()) {
      std::stringstream ss;
      ss << "TensorShape mismatch in bind_dims(). "
         << "Tensor shape: " << sizes.size() << ", dims: " << dims.size();
      throw std::runtime_error(ss.str());
    }

    for (size_t i = 0; i < dims.size(); i++) {
      auto& dim = dims.at(i);
      if (!dim.impl_->size) {
        dim.impl_->size.reset(new size_t(sizes[i]));
      } else if (sizes[i] != *dim.impl_->size) {
        std::stringstream ss;
        ss << "bind_dims() mismatch on dim " << i << ". "
           << "Required: " << *dim.impl_->size << ", Actual: " << sizes[i];
        throw std::runtime_error(ss.str());
      }
    }
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
Tensor NamedTensorOutput(const std::string& name, Ts... dims) {
  std::vector<TensorDim> vec;
  details::into_vector(&vec, dims...);
  return Tensor{name, vec};
}

template <typename... Ts>
Tensor TensorOutput(Ts... dims) {
  std::vector<TensorDim> vec;
  details::into_vector(&vec, dims...);
  return Tensor{vec};
}

Tensor Call(const std::string& fn, const std::vector<Tensor>& args);

template <typename... Ts>
Tensor Call(const std::string& fn, Ts... args) {
  std::vector<Tensor> vec;
  details::into_vector(&vec, std::forward<Ts>(args)...);
  return Call(fn, vec);
}

inline Tensor as_float(const Tensor& x, size_t bit_size) { return Call("as_float", x, static_cast<int64_t>(bit_size)); }

inline Tensor as_int(const Tensor& x, size_t bit_size) { return Call("as_int", x, static_cast<int64_t>(bit_size)); }

inline Tensor as_uint(const Tensor& x, size_t bit_size) { return Call("as_uint", x, static_cast<int64_t>(bit_size)); }

// inline Tensor element(const Tensor& x) { return Call("element", {x}); } // TODO: tuple

inline Tensor cos(const Tensor& x) { return Call("cos", x); }

inline Tensor exp(const Tensor& x) { return Call("exp", x); }

inline Tensor gather(const Tensor& x, const Tensor& y) { return Call("gather", x, y); }

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

inline Tensor reshape(const Tensor& x, const TensorShape& shape) {
  std::vector<Tensor> args = {x};
  for (size_t i = 0; i < shape.rank(); i++) {
    args.emplace_back(static_cast<int64_t>(shape.size_at(i)));
  }
  return Call("reshape", args);
}

inline Tensor scatter(const Tensor& x, const Tensor& y, const Tensor& z) { return Call("scatter", x, y, z); }

inline Tensor select(const Tensor& cond, const Tensor& true_case, const Tensor& false_case) {
  return Call("cond", cond, true_case, false_case);
}

inline Tensor shape(const Tensor& x) { return Call("shape", x); }

inline Tensor sigmoid(const Tensor& x) { return Call("sigmoid", x); }

inline Tensor sin(const Tensor& x) { return Call("sin", x); }

inline Tensor sqrt(const Tensor& x) { return Call("sqrt", x); }

inline Tensor tan(const Tensor& x) { return Call("tan", x); }

inline Tensor tanh(const Tensor& x) { return Call("tanh", x); }

inline std::ostream& operator<<(std::ostream& os, const TensorShape& shape) {
  os << shape.str();
  return os;
}

class TensorFriend {
 public:
  static tile_program* evaluate(const std::string& name, const std::vector<Tensor>& tensors) {
    std::vector<tile_expr*> exprs;
    for (const auto& tensor : tensors) {
      exprs.emplace_back(tensor.impl_->ptr.get());
    }
    return ffi::call<tile_program*>(tile_program_evaluate, name.c_str(), exprs.size(), exprs.data());
  }

  static TensorIndex MakePolyOp(tile_poly_op op, const std::vector<TensorIndex>& args) {
    std::vector<tile_poly_expr*> operands;
    for (const auto& arg : args) {
      operands.push_back(arg.impl_->ptr.get());
    }
    auto impl = std::make_shared<TensorIndex::Impl>();
    impl->ptr = details::make_tile_poly_expr(  //
        ffi::call<tile_poly_expr*>(            //
            tile_poly_expr_op,                 //
            op,                                //
            operands.size(),                   //
            operands.data()));
    return TensorIndex(impl);
  }

  static TensorIndex MakeMixedPolyBinaryOp(tile_poly_op op, const TensorIndex& idx, const TensorDim& dim,
                                           bool lhs_first) {
    if (!dim.impl_->size) {
      throw std::runtime_error("Undefined dimension.");
    }
    std::vector<tile_poly_expr*> operands;
    auto dim_ptr = ffi::call<tile_poly_expr*>(tile_poly_expr_literal, *dim.impl_->size);
    if (lhs_first) {
      operands.emplace_back(idx.impl_->ptr.get());
      operands.emplace_back(dim_ptr);
    } else {
      operands.emplace_back(dim_ptr);
      operands.emplace_back(idx.impl_->ptr.get());
    }
    auto impl = std::make_shared<TensorIndex::Impl>();
    impl->ptr = details::make_tile_poly_expr(  //
        ffi::call<tile_poly_expr*>(            //
            tile_poly_expr_op,                 //
            op,                                //
            operands.size(),                   //
            operands.data()));
    return TensorIndex(impl);
  }

  static TensorDim DimOp(tile_poly_op op, const std::vector<TensorDim>& args) {
    std::vector<size_t> sizes;
    for (const auto& arg : args) {
      if (!arg.impl_->size) {
        throw std::runtime_error("Undefined dimension.");
      }
      sizes.emplace_back(*arg.impl_->size);
    }
    switch (op) {
      case TILE_POLY_OP_NEG:
        return TensorDim{-sizes[0]};
      case TILE_POLY_OP_ADD:
        return TensorDim{sizes[0] + sizes[1]};
      case TILE_POLY_OP_SUB:
        return TensorDim{sizes[0] - sizes[1]};
      case TILE_POLY_OP_MUL:
        return TensorDim{sizes[0] * sizes[1]};
      case TILE_POLY_OP_DIV:
        return TensorDim{sizes[0] / sizes[1]};
      default:
        throw std::runtime_error("Invalid poly op");
    }
  }

  static IndexedTensor ComboParts(tile_combo_op op, const std::vector<const IndexedTensor*>& args) {
    auto impl = std::make_unique<IndexedTensor::Impl>();
    impl->nary.reset(new IndexedTensor::ComboParts);
    impl->nary->op = op;
    for (const auto& arg : args) {
      impl->nary->args.emplace_back(arg->impl_->unary.get());
    }
    return IndexedTensor{std::move(impl)};
  }

  static Tensor Call(const std::string& fn, const std::vector<Tensor>& args) {
    std::vector<tile_expr*> ptrs(args.size());
    for (size_t i = 0; i < args.size(); i++) {
      ptrs[i] = args[i].impl_->ptr.get();
    }
    auto impl = std::make_unique<Tensor::Impl>();
    impl->ptr = details::make_tile_expr(  //
        ffi::call<tile_expr*>(            //
            tile_expr_call,               //
            fn.c_str(),                   //
            ptrs.size(),                  //
            ptrs.data()));
    return Tensor{std::move(impl)};
  }
};

inline Program::Program(const std::string& name, const std::vector<Tensor>& tensors)
    : ptr_(details::make_tile_program(TensorFriend::evaluate(name, tensors))) {}

inline TensorIndex TensorIndex::operator-() const { return TensorFriend::MakePolyOp(TILE_POLY_OP_NEG, {*this}); }

#define TILE_CC_DEFINE_TENSOR_IDXDIM_BINARY_OPS(_op_, _poly_op_)                     \
  inline TensorIndex operator _op_(const TensorIndex& lhs, const TensorIndex& rhs) { \
    return TensorFriend::MakePolyOp(_poly_op_, {lhs, rhs});                          \
  }                                                                                  \
  inline TensorIndex operator _op_(const TensorIndex& lhs, size_t rhs) {             \
    return TensorFriend::MakePolyOp(_poly_op_, {lhs, TensorIndex{rhs}});             \
  }                                                                                  \
  inline TensorIndex operator _op_(size_t lhs, const TensorIndex& rhs) {             \
    return TensorFriend::MakePolyOp(_poly_op_, {TensorIndex{lhs}, rhs});             \
  }                                                                                  \
  inline TensorIndex operator _op_(const TensorIndex& lhs, const TensorDim& rhs) {   \
    return TensorFriend::MakeMixedPolyBinaryOp(_poly_op_, lhs, rhs, true);           \
  }                                                                                  \
  inline TensorIndex operator _op_(const TensorDim& lhs, const TensorIndex& rhs) {   \
    return TensorFriend::MakeMixedPolyBinaryOp(_poly_op_, rhs, lhs, false);          \
  }                                                                                  \
  inline TensorDim operator _op_(const TensorDim& lhs, const TensorDim& rhs) {       \
    return TensorFriend::DimOp(_poly_op_, {lhs, rhs});                               \
  }                                                                                  \
  inline TensorDim operator _op_(size_t lhs, const TensorDim& rhs) {                 \
    return TensorFriend::DimOp(_poly_op_, {TensorDim{lhs}, rhs});                    \
  }                                                                                  \
  inline TensorDim operator _op_(const TensorDim& lhs, size_t rhs) {                 \
    return TensorFriend::DimOp(_poly_op_, {lhs, TensorDim{rhs}});                    \
  }

TILE_CC_DEFINE_TENSOR_IDXDIM_BINARY_OPS(+, TILE_POLY_OP_ADD);
TILE_CC_DEFINE_TENSOR_IDXDIM_BINARY_OPS(-, TILE_POLY_OP_SUB);
TILE_CC_DEFINE_TENSOR_IDXDIM_BINARY_OPS(*, TILE_POLY_OP_MUL);
TILE_CC_DEFINE_TENSOR_IDXDIM_BINARY_OPS(/, TILE_POLY_OP_DIV);

inline Tensor Tensor::operator-() const { return Call("neg", {*this}); }
inline Tensor Tensor::operator~() const { return Call("bit_not", {*this}); }

#define TILE_CC_DEFINE_TENSOR_BINARY_OPS(_op_, _fn_)                                                   \
  inline Tensor operator _op_(const Tensor& lhs, const Tensor& rhs) { return Call(_fn_, lhs, rhs); }   \
  inline Tensor operator _op_(const Tensor& lhs, int rhs) { return Call(_fn_, lhs, Tensor{rhs}); }     \
  inline Tensor operator _op_(const Tensor& lhs, int64_t rhs) { return Call(_fn_, lhs, Tensor{rhs}); } \
  inline Tensor operator _op_(const Tensor& lhs, double rhs) { return Call(_fn_, lhs, Tensor{rhs}); }  \
  inline Tensor operator _op_(int lhs, const Tensor& rhs) { return Call(_fn_, Tensor{lhs}, rhs); }     \
  inline Tensor operator _op_(int64_t lhs, const Tensor& rhs) { return Call(_fn_, Tensor{lhs}, rhs); } \
  inline Tensor operator _op_(double lhs, const Tensor& rhs) { return Call(_fn_, Tensor{lhs}, rhs); }

TILE_CC_DEFINE_TENSOR_BINARY_OPS(+, "add");
TILE_CC_DEFINE_TENSOR_BINARY_OPS(-, "sub");
TILE_CC_DEFINE_TENSOR_BINARY_OPS(*, "mul");
TILE_CC_DEFINE_TENSOR_BINARY_OPS(/, "div");
TILE_CC_DEFINE_TENSOR_BINARY_OPS(==, "cmp_eq");
TILE_CC_DEFINE_TENSOR_BINARY_OPS(!=, "cmp_ne");
TILE_CC_DEFINE_TENSOR_BINARY_OPS(<, "cmp_lt");
TILE_CC_DEFINE_TENSOR_BINARY_OPS(>, "cmp_gt");
TILE_CC_DEFINE_TENSOR_BINARY_OPS(<=, "cmp_le");
TILE_CC_DEFINE_TENSOR_BINARY_OPS(>=, "cmp_ge");
TILE_CC_DEFINE_TENSOR_BINARY_OPS(<<, "bit_left");
TILE_CC_DEFINE_TENSOR_BINARY_OPS(>>, "bit_right");
TILE_CC_DEFINE_TENSOR_BINARY_OPS(&, "bit_and");
TILE_CC_DEFINE_TENSOR_BINARY_OPS(|, "bit_or");
TILE_CC_DEFINE_TENSOR_BINARY_OPS(^, "bit_xor");

inline void IndexedTensor::Impl::MakeContraction(tile_agg_op agg_op, const IndexedTensor& rhs) {
  tile_combo_op combo_op;
  std::vector<tile_expr*> inputs;
  if (rhs.impl_->unary) {
    // unary op: TensorSpec
    combo_op = TILE_COMBO_OP_NONE;
    inputs.emplace_back(rhs.impl_->unary.get());
  } else if (rhs.impl_->nary) {
    // binary/ternary op: ComboParts
    combo_op = rhs.impl_->nary->op;
    inputs = rhs.impl_->nary->args;
  } else {
    throw std::runtime_error("Invalid impl");
  }
  src->impl_->ptr = details::make_tile_expr(  //
      ffi::call<tile_expr*>(                  //
          tile_expr_contraction,              //
          agg_op,                             //
          combo_op,                           //
          unary.get(),                        //
          inputs.size(),                      //
          inputs.data(),                      //
          src->impl_->name.c_str()));
}

// Represents a combo_op of COND in a contraction
inline IndexedTensor cond(const IndexedTensor& lhs, const IndexedTensor& rhs, const IndexedTensor& true_case) {
  return TensorFriend::ComboParts(TILE_COMBO_OP_COND, {&lhs, &rhs, &true_case});
}

inline IndexedTensor IndexedTensor::operator+(const IndexedTensor& rhs) const {  //
  return TensorFriend::ComboParts(TILE_COMBO_OP_ADD, {this, &rhs});
}

inline IndexedTensor IndexedTensor::operator*(const IndexedTensor& rhs) const {  //
  return TensorFriend::ComboParts(TILE_COMBO_OP_MUL, {this, &rhs});
}

inline IndexedTensor IndexedTensor::operator==(const IndexedTensor& rhs) const {  //
  return TensorFriend::ComboParts(TILE_COMBO_OP_EQ, {this, &rhs});
}

inline Tensor Call(const std::string& fn, const std::vector<Tensor>& args) { return TensorFriend::Call(fn, args); }

}  // namespace edsl
}  // namespace plaidml
}  // namespace vertexai
