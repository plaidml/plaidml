// Copyright 2019 Intel Corporation.

#include "plaidml2/op/ops.h"

#include "base/util/logging.h"

using namespace plaidml::edsl;  // NOLINT

namespace plaidml {
namespace op {

struct AggregationAxes {
  std::vector<TensorIndex> src_idxs;
  std::vector<TensorIndex> dst_idxs;
  std::vector<TensorDim> src_dims;
  std::vector<TensorDim> dst_dims;
  std::set<size_t> axes;

  AggregationAxes(size_t ndims, const Value& in_axes, bool keepdims) : src_idxs(ndims), src_dims(ndims) {
    if (in_axes.is_none()) {
      axes = {ndims - 1};
    }
    if (in_axes.is_tuple()) {
      for (const auto& axis : in_axes.as_tuple()) {
        axes.insert(axis.as_int());
      }
    } else if (in_axes.is_int()) {
      auto axis = in_axes.as_int();
      if (axis < 0) {
        axis = ndims + axis;
      }
      axes = {static_cast<size_t>(axis)};
    } else {
      throw std::runtime_error("Invalid Value type for AggregationAxes: in_axes");
    }

    if (keepdims) {
      dst_idxs = src_idxs;
      dst_dims = src_dims;
      for (auto axis : axes) {
        dst_idxs[axis] = TensorIndex();
        dst_dims[axis] = TensorDim{1};
      }
    } else {
      for (size_t i = 0; i < ndims; i++) {
        if (!axes.count(i)) {
          dst_idxs.push_back(src_idxs[i]);
          dst_dims.push_back(src_dims[i]);
        }
      }
    }
  }
};

Value mean(const Value& value) {
  IVLOG(1, "mean");
  auto args = value.as_tuple();
  if (args.size() != 3) {
    throw std::runtime_error("mean expects 3 arguments");
  }

  auto I = args[0].as_tensor();
  auto I_shape = I.shape();
  if (I_shape.ndims() == 0) {
    return Value{I};
  }

  // if (I_shape.dtype() == PLAIDML_DATA_BOOLEAN) {
  //   I = cast(I, floatx());
  // }

  auto axes = args[1];
  if (axes.is_tuple() && axes.as_tuple().empty()) {
    return Value{I};
  }

  auto keepdims = args[2].as_int();

  AggregationAxes agg(I_shape.ndims(), axes, keepdims);

  I.bind_dims(agg.src_dims);
  auto SO = TensorOutput(agg.dst_dims);
  SO(agg.dst_idxs) += I(agg.src_idxs);
  auto denom = Tensor{1};
  for (const auto& axis : agg.axes) {
    denom = denom * agg.src_dims[axis];
  }
  return Value{SO / denom};
}

Value sum(const Value& value) {
  IVLOG(1, "sum");
  auto args = value.as_tuple();
  if (args.size() != 3) {
    throw std::runtime_error("mean expects 3 arguments");
  }

  auto I = args[0].as_tensor();
  auto I_shape = I.shape();
  if (I_shape.ndims() == 0) {
    return Value{I};
  }

  // if (I_shape.dtype() == PLAIDML_DATA_BOOLEAN) {
  //   I = cast(I, floatx());
  // }

  auto axes = args[1];
  if (axes.is_tuple() && axes.as_tuple().empty()) {
    return Value{I};
  }

  auto keepdims = args[2].as_int();

  AggregationAxes agg(I_shape.ndims(), axes, keepdims);

  I.bind_dims(agg.src_dims);
  auto O = TensorOutput(agg.dst_dims);
  O(agg.dst_idxs) += I(agg.src_idxs);

  return Value{O};
}

[[gnu::unused]] auto init = []() {
  auto registry = OperationRegistry::Instance();
  registry->Register("mean", mean);
  registry->Register("sum", sum);
  return 0;
}();

}  // namespace op
}  // namespace plaidml
