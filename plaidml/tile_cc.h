#pragma once

#include <algorithm>
#include <memory>
#include <string>
#include <utility>
#include <vector>

#include "tile/base/shape.h"
#include "tile/lang/ops.h"

namespace vertexai {
namespace plaidml {
namespace tile_cc {

class Access;
class Index;
class Tensor;

class IndexIterator {
 public:
  IndexIterator() = default;
  explicit IndexIterator(Index* index) : index_(index) {}

  IndexIterator& operator++() {
    index_ = nullptr;
    return *this;
  }

  bool operator!=(const IndexIterator& other) { return index_ != other.index_; }
  Index& operator*() { return *index_; }

 private:
  Index* index_ = nullptr;
};

class Constraint {
  friend class Index;

 public:
  Constraint operator&&(const Constraint& rhs) const { return Constraint(); }
  operator bool() const { return true; }

 private:
  Constraint() = default;
};

class Index {
  friend class Tensor;
  friend struct ConstraintCollector;
  struct Impl;

 public:
  Index();
  ~Index();

  Index(size_t value);  // NOLINT

  IndexIterator begin() { return IndexIterator(this); }
  IndexIterator end() { return IndexIterator{}; }

  Index operator-() const;
  Index operator+(const Index& rhs) const;
  Index operator-(const Index& rhs) const;
  Index operator*(const Index& rhs) const;
  Index operator/(const Index& rhs) const;

  Constraint operator<(size_t rhs) const;

 private:
  std::shared_ptr<Impl> impl_;
  explicit Index(std::unique_ptr<Impl> impl);
};

template <typename T>
Index operator+(T lhs, const Index& rhs) {
  return rhs + lhs;
}

template <typename T>
Index operator*(T lhs, const Index& rhs) {
  return rhs * lhs;
}

class Access {
  friend class Tensor;
  friend Access cond(const Access&, const Access&, const Access&);
  struct Impl;

 public:
  ~Access();

  // Movable constructor
  Access(Access&& rhs) noexcept;

  // Represents an aggregation_op of SUM in a contraction
  Access& operator+=(const Access& rhs);

  // Represents an aggregation_op of PROD in a contraction
  Access& operator*=(const Access& rhs);

  // Represents an aggregation_op of MAX in a contraction
  Access& operator>=(const Access& rhs);

  // Represents an aggregation_op of MIN in a contraction
  Access& operator<=(const Access& rhs);

  // Represents an aggregation_op of ASSIGN in a contraction
  Access& operator=(const Access& rhs);

  // Represents a combo_op of PLUS in a contraction
  Access operator+(const Access& rhs) const;

  // Represents a combo_op of MULTIPLY in a contraction
  Access operator*(const Access& rhs) const;

  // Represents a combo_op of EQ in a contraction
  Access operator==(const Access& rhs) const;

  const Impl* impl() const;

 private:
  std::unique_ptr<Impl> impl_;
  explicit Access(std::unique_ptr<Impl> impl);
};

inline Access sum(Access lhs, const Access& rhs) { return std::move(lhs += rhs); }

inline Access product(Access lhs, const Access& rhs) { return std::move(lhs *= rhs); }

inline Access max(Access lhs, const Access& rhs) { return std::move(lhs >= rhs); }

inline Access min(Access lhs, const Access& rhs) { return std::move(lhs <= rhs); }

inline Access assign(Access lhs, const Access& rhs) { return std::move(lhs = rhs); }

// Represents a combo_op of COND in a contraction
Access cond(const Access& cond_lhs, const Access& cond_rhs, const Access& true_case);

class Tensor {
  friend class Access;
  friend Tensor Call(const std::string& fn, const std::vector<Tensor>& args);
  struct Impl;

 public:
  Tensor();
  ~Tensor();

  Tensor(int value);      // NOLINT
  Tensor(int64_t value);  // NOLINT
  Tensor(double value);   // NOLINT
  explicit Tensor(const tile::TensorShape& shape);

  // Copyable
  Tensor(const Tensor& rhs);
  Tensor& operator=(const Tensor& rhs);

  Access operator()(const std::vector<Index>& idxs, const std::vector<size_t>& sizes);
  Access operator()(const std::vector<Index>& idxs) const;
  size_t operator[](const size_t dim) const;

  // Represents an eltwise negation
  Tensor operator-() const;

  // Represents an eltwise bit_not
  Tensor operator~() const;

  // Represents an eltwise addition
  Tensor operator+(const Tensor& rhs) const;

  // Represents an eltwise subtraction
  Tensor operator-(const Tensor& rhs) const;

  // Represents an eltwise multiplication
  Tensor operator*(const Tensor& rhs) const;

  // Represents an eltwise division
  Tensor operator/(const Tensor& rhs) const;

  // Represents an eltwise cmp_eq
  Tensor operator==(const Tensor& rhs) const;

  // Represents an eltwise cmp_ne
  Tensor operator!=(const Tensor& rhs) const;

  // Represents an eltwise cmp_lt
  Tensor operator<(const Tensor& rhs) const;

  // Represents an eltwise cmp_gt
  Tensor operator>(const Tensor& rhs) const;

  // Represents an eltwise cmp_le
  Tensor operator<=(const Tensor& rhs) const;

  // Represents an eltwise cmp_ge
  Tensor operator>=(const Tensor& rhs) const;

  // Represents an eltwise bit_left
  Tensor operator<<(const Tensor& rhs) const;

  // Represents an eltwise bit_right
  Tensor operator>>(const Tensor& rhs) const;

  // Represents an eltwise bit_and
  Tensor operator&(const Tensor& rhs) const;

  // Represents an eltwise bit_or
  Tensor operator|(const Tensor& rhs) const;

  // Represents an eltwise bit_xor
  Tensor operator^(const Tensor& rhs) const;

  // Enable no_defract on a contraction
  Tensor& no_defract();

  // Set use_default on a contraction
  Tensor& use_default(const Tensor& rhs);

  const Impl* impl() const;
  tile::TensorShape shape() const;

 private:
  explicit Tensor(std::unique_ptr<Impl> impl);

 private:
  std::unique_ptr<Impl> impl_;
};

template <typename T>
Tensor operator+(T lhs, const Tensor& rhs) {
  return rhs + lhs;
}

template <typename T>
Tensor operator*(T lhs, const Tensor& rhs) {
  return rhs * lhs;
}

Tensor Call(const std::string& fn, const std::vector<Tensor>& args);

inline Tensor as_float(const Tensor& x, size_t bit_size) {
  return Call("as_float", {x, static_cast<int64_t>(bit_size)});
}

inline Tensor as_int(const Tensor& x, size_t bit_size) { return Call("as_int", {x, static_cast<int64_t>(bit_size)}); }

inline Tensor as_uint(const Tensor& x, size_t bit_size) { return Call("as_uint", {x, static_cast<int64_t>(bit_size)}); }

// inline Tensor element(const Tensor& x) { return Call("element", {x}); } // TODO: tuple

inline Tensor exp(const Tensor& x) { return Call("exp", {x}); }

inline Tensor gather(const Tensor& x, const Tensor& y) { return Call("gather", {x, y}); }

inline Tensor index(const Tensor& x, size_t axis) { return Call("index", {x, static_cast<int64_t>(axis)}); }

inline Tensor prng_state(const Tensor& x) { return Call("prng_state", {x}); }

inline Tensor prng_step(const Tensor& x, const std::vector<size_t>& sizes) {
  std::vector<Tensor> args = {x};
  for (const auto& size : sizes) {
    args.emplace_back(static_cast<int64_t>(size));
  }
  return Call("prng_step", args);
}

inline Tensor prng_value(const Tensor& x) { return Call("prng_value", {x}); }

inline Tensor reshape(const Tensor& x, const tile::TensorShape& shape) {
  std::vector<Tensor> args = {x};
  for (const auto& dim : shape.dims) {
    args.emplace_back(static_cast<int64_t>(dim.size));
  }
  return Call("reshape", args);
}

inline Tensor scatter(const Tensor& x, const Tensor& y, const Tensor& z) { return Call("scatter", {x, y, z}); }

inline Tensor select(const Tensor& cond, const Tensor& true_case, const Tensor& false_case) {
  return Call("cond", {cond, true_case, false_case});
}

inline Tensor shape(const Tensor& x) { return Call("shape", {x}); }

inline Tensor sqrt(const Tensor& x) { return Call("sqrt", {x}); }

tile::lang::Program Evaluate(const std::vector<Tensor>& vars);

}  // namespace tile_cc
}  // namespace plaidml
}  // namespace vertexai
