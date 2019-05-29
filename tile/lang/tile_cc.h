#pragma once

#include <algorithm>
#include <memory>
#include <string>
#include <utility>
#include <vector>

#include "tile/base/shape.h"
#include "tile/lang/compose.h"

namespace vertexai {
namespace tile {
namespace lang {

class IndexedTensor;
class Tensor;
class TensorDim;
class TensorFriend;
class TensorIndex;

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
  struct Impl;

 public:
  TensorDim();
  explicit TensorDim(size_t value);

  TensorDim operator-() const;

 private:
  std::shared_ptr<Impl> impl_;
};

class TensorIndex {
  friend class Tensor;
  friend class TensorFriend;
  struct Impl;

 public:
  TensorIndex();

  explicit TensorIndex(size_t value);
  explicit TensorIndex(const std::string& name);

  TensorIndexIterator begin() { return TensorIndexIterator(this); }
  TensorIndexIterator end() { return TensorIndexIterator{}; }

  TensorIndex operator-() const;

  Constraint operator<(size_t rhs) const;
  Constraint operator<(const TensorDim& rhs) const;

 private:
  std::shared_ptr<Impl> impl_;
  explicit TensorIndex(std::unique_ptr<Impl> impl);
};

#define TILE_CC_DECLARE_TENSOR_IDXDIM_BINARY_OPS(_op_)                       \
  TensorDim operator _op_(const TensorDim& lhs, const TensorDim& rhs);       \
  TensorDim operator _op_(const TensorDim& lhs, size_t rhs);                 \
  TensorDim operator _op_(size_t lhs, const TensorDim& rhs);                 \
  TensorIndex operator _op_(const TensorIndex& lhs, const TensorIndex& rhs); \
  TensorIndex operator _op_(const TensorIndex& lhs, size_t rhs);             \
  TensorIndex operator _op_(size_t lhs, const TensorIndex& rhs);             \
  TensorIndex operator _op_(const TensorDim& lhs, const TensorIndex& rhs);   \
  TensorIndex operator _op_(const TensorIndex& lhs, const TensorDim& rhs);

TILE_CC_DECLARE_TENSOR_IDXDIM_BINARY_OPS(+);
TILE_CC_DECLARE_TENSOR_IDXDIM_BINARY_OPS(-);
TILE_CC_DECLARE_TENSOR_IDXDIM_BINARY_OPS(*);
TILE_CC_DECLARE_TENSOR_IDXDIM_BINARY_OPS(/);

class IndexedTensor {
  friend class Tensor;
  friend class TensorFriend;
  struct Impl;

 public:
  ~IndexedTensor();

  // Movable constructor
  IndexedTensor(IndexedTensor&& rhs) noexcept;

  // Represents an aggregation_op of SUM in a contraction
  IndexedTensor& operator+=(const IndexedTensor& rhs);

  // Represents an aggregation_op of PROD in a contraction
  IndexedTensor& operator*=(const IndexedTensor& rhs);

  // Represents an aggregation_op of MAX in a contraction
  IndexedTensor& operator>=(const IndexedTensor& rhs);

  // Represents an aggregation_op of MIN in a contraction
  IndexedTensor& operator<=(const IndexedTensor& rhs);

  // Represents an aggregation_op of ASSIGN in a contraction
  IndexedTensor& operator=(const IndexedTensor& rhs);

  // Represents a combo_op of PLUS in a contraction
  IndexedTensor operator+(const IndexedTensor& rhs) const;

  // Represents a combo_op of MULTIPLY in a contraction
  IndexedTensor operator*(const IndexedTensor& rhs) const;

  // Represents a combo_op of EQ in a contraction
  IndexedTensor operator==(const IndexedTensor& rhs) const;

 private:
  std::unique_ptr<Impl> impl_;
  explicit IndexedTensor(std::unique_ptr<Impl> impl);
};

inline IndexedTensor sum(IndexedTensor lhs, const IndexedTensor& rhs) { return std::move(lhs += rhs); }

inline IndexedTensor product(IndexedTensor lhs, const IndexedTensor& rhs) { return std::move(lhs *= rhs); }

inline IndexedTensor max(IndexedTensor lhs, const IndexedTensor& rhs) { return std::move(lhs >= rhs); }

inline IndexedTensor min(IndexedTensor lhs, const IndexedTensor& rhs) { return std::move(lhs <= rhs); }

inline IndexedTensor assign(IndexedTensor lhs, const IndexedTensor& rhs) { return std::move(lhs = rhs); }

// Represents a combo_op of COND in a contraction
IndexedTensor cond(const IndexedTensor& cond_lhs, const IndexedTensor& cond_rhs, const IndexedTensor& true_case);

template <typename T>
void IntoVector(std::vector<T>*) {}

template <typename T, typename Head, typename... Tail>
void IntoVector(std::vector<T>* into, Head&& head, Tail&&... tail) {
  into->emplace_back(std::forward<Head>(head));
  IntoVector(into, std::forward<Tail>(tail)...);
}

template <typename... Ts>
std::vector<TensorDim> MakeTensorDims(Ts... dims) {
  std::vector<TensorDim> vec;
  IntoVector(&vec, std::forward<Ts>(dims)...);
  return vec;
}

class Tensor {
  friend class IndexedTensor;
  friend class TensorFriend;
  struct Impl;

 public:
  Tensor();
  ~Tensor();

  explicit Tensor(int value);
  explicit Tensor(int64_t value);
  explicit Tensor(double value);

  explicit Tensor(const tile::TensorShape& shape);
  explicit Tensor(const std::vector<TensorDim>& dims);
  explicit Tensor(const std::initializer_list<TensorDim>& dims);

  explicit Tensor(const std::string& name);
  Tensor(const std::string& name, const tile::TensorShape& shape);
  Tensor(const std::string& name, const std::vector<TensorDim>& dims);
  Tensor(const std::string& name, const std::initializer_list<TensorDim>& dims);

  template <typename... Ts>
  Tensor(const std::string& name, Ts... dims) : Tensor(name, MakeTensorDims(dims...)) {}

  // Copyable
  Tensor(const Tensor& rhs);
  Tensor& operator=(const Tensor& rhs);

  IndexedTensor operator()(const std::vector<TensorIndex>& idxs) const;

  IndexedTensor operator()(const std::initializer_list<TensorIndex>& idxs) const {
    return operator()(std::vector<TensorIndex>{idxs});
  }

  template <typename... Ts>
  IndexedTensor operator()(Ts... idxs) const {
    std::vector<TensorIndex> vec;
    IntoVector(&vec, std::forward<Ts>(idxs)...);
    return operator()(vec);
  }

  // Represents an eltwise negation
  Tensor operator-() const;

  // Represents an eltwise bit_not
  Tensor operator~() const;

  // Enable no_defract on a contraction
  Tensor& no_defract();

  // Set use_default on a contraction
  Tensor& use_default(const Tensor& rhs);

  // Return the tensor's shape
  tile::TensorShape shape() const;

  // Return the size of the tensor's shape at the specified dimension.
  size_t dims(const size_t dim) const;

  // Verify that the specified dims match the dims of this tensor.
  void bind_dims(const std::vector<TensorDim>& dims) const;

  template <typename... Ts>
  void bind_dims(Ts... dims) const {
    std::vector<TensorDim> vec;
    IntoVector(&vec, std::forward<Ts>(dims)...);
    bind_dims(vec);
  }

 private:
  explicit Tensor(std::unique_ptr<Impl> impl);

 private:
  std::unique_ptr<Impl> impl_;
};

template <typename... Ts>
Tensor TensorOutput(Ts... dims) {
  std::vector<TensorDim> vec;
  IntoVector(&vec, dims...);
  return Tensor{vec};
}

#define TILE_CC_DECLARE_TENSOR_BINARY_OPS(_op_)               \
  Tensor operator _op_(const Tensor& lhs, const Tensor& rhs); \
  Tensor operator _op_(const Tensor& lhs, int rhs);           \
  Tensor operator _op_(const Tensor& lhs, int64_t rhs);       \
  Tensor operator _op_(const Tensor& lhs, double rhs);        \
  Tensor operator _op_(int lhs, const Tensor& rhs);           \
  Tensor operator _op_(int64_t lhs, const Tensor& rhs);       \
  Tensor operator _op_(double lhs, const Tensor& rhs);

// Represents an eltwise addition
TILE_CC_DECLARE_TENSOR_BINARY_OPS(+);

// Represents an eltwise subtraction
TILE_CC_DECLARE_TENSOR_BINARY_OPS(-);

// Represents an eltwise multiplication
TILE_CC_DECLARE_TENSOR_BINARY_OPS(*);

// Represents an eltwise division
TILE_CC_DECLARE_TENSOR_BINARY_OPS(/);

// Represents an eltwise cmp_eq
TILE_CC_DECLARE_TENSOR_BINARY_OPS(==);

// Represents an eltwise cmp_ne
TILE_CC_DECLARE_TENSOR_BINARY_OPS(!=);

// Represents an eltwise cmp_lt
TILE_CC_DECLARE_TENSOR_BINARY_OPS(<);  // NOLINT

// Represents an eltwise cmp_gt
TILE_CC_DECLARE_TENSOR_BINARY_OPS(>);  // NOLINT

// Represents an eltwise cmp_le
TILE_CC_DECLARE_TENSOR_BINARY_OPS(<=);

// Represents an eltwise cmp_ge
TILE_CC_DECLARE_TENSOR_BINARY_OPS(>=);

// Represents an eltwise bit_left
TILE_CC_DECLARE_TENSOR_BINARY_OPS(<<);

// Represents an eltwise bit_right
TILE_CC_DECLARE_TENSOR_BINARY_OPS(>>);

// Represents an eltwise bit_and
TILE_CC_DECLARE_TENSOR_BINARY_OPS(&);

// Represents an eltwise bit_or
TILE_CC_DECLARE_TENSOR_BINARY_OPS(|);

// Represents an eltwise bit_xor
TILE_CC_DECLARE_TENSOR_BINARY_OPS(^);

Tensor Call(const std::string& fn, const std::vector<Tensor>& args);

template <typename... Ts>
Tensor Call(const std::string& fn, Ts... args) {
  std::vector<Tensor> vec;
  IntoVector(&vec, std::forward<Ts>(args)...);
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

inline Tensor reshape(const Tensor& x, const tile::TensorShape& shape) {
  std::vector<Tensor> args = {x};
  for (const auto& dim : shape.dims) {
    args.emplace_back(static_cast<int64_t>(dim.size));
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

tile::lang::RunInfo Evaluate(const std::string& name, const std::vector<Tensor>& vars);

}  // namespace lang
}  // namespace tile
}  // namespace vertexai
