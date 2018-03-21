#pragma once

#include <map>
#include <string>
#include <vector>

#include "base/util/transfer_object.h"
#include "tile/lang/ops.h"
#include "tile/lang/shape.h"
#include "tile/lang/type.h"

namespace vertexai {
namespace tile {
namespace lang {

struct FlatTensorAccess {
  DataType type = DataType::INVALID;
  uint64_t vector = 1;  // Vector width of this tensor
  int64_t offset = 0;
  uint64_t global_index_limit = 0;
  std::vector<int64_t> strides;
  uint64_t elem_size() const { return byte_width(type) * vector; }
  TRANSFER_OBJECT {
    VERSION(0);
    FIELD(type);
    FIELD(vector);
    FIELD(offset);
    FIELD(global_index_limit);
    FIELD(strides);
  }
};

struct FlatConstraint {
  std::vector<int64_t> lhs;
  int64_t rhs;
  TRANSFER_OBJECT {
    VERSION(0);
    FIELD(lhs);
    FIELD(rhs);
  }
};

struct FlatPostOpInput {
  std::string name;
  FlatTensorAccess access;
  Binding binding;
};

struct FlatContraction {
  FlatContraction() = default;
  explicit FlatContraction(const Contraction& c);

  // Tile cache key members
  std::vector<uint64_t> ranges;
  std::vector<FlatTensorAccess> access;
  std::vector<FlatConstraint> constraints;
  DataType agg_type = DataType::BOOLEAN;
  uint64_t agg_vec = 1;
  CombinationOp comb_op;
  AggregationOp agg_op;
  bool generate_contraction = true;

  std::string TileKeyString() const;

  // non-primary key members
  std::vector<std::string> inputs;
  std::vector<std::string> names;
  std::string comments;
  std::string output;  // The contraction output -- not necessarily written.

  // Full cache primary key (names replaced to protect the innocent)
  std::vector<Op> post_ops;
  std::vector<FlatPostOpInput> post_op_inputs;  // Additional inputs for the post_ops
  std::vector<std::string> kernel_outputs;      // Outputs written by the kernel.

  std::string CacheKeyString(const Bindings& vars) const;

  std::string toString() const;
};

// Require Contraction to be in reduced form
FlatContraction Flatten(const Contraction& c, const std::vector<TensorShape>& shapes);

inline std::string to_string(const FlatContraction& fc) { return fc.toString(); }

inline MAKE_LOGGABLE(FlatContraction, f, os) {
  os << to_string(f);
  return os;
}

}  // namespace lang
}  // namespace tile
}  // namespace vertexai
