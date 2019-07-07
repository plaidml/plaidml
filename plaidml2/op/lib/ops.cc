// Copyright 2019 Intel Corporation.

#include "plaidml2/op/lib/ops.h"

#include <boost/format.hpp>

#include "base/util/logging.h"
#include "plaidml2/op/op.h"

using namespace plaidml::edsl;  // NOLINT

namespace plaidml {
namespace op {
namespace lib {

struct AggregationAxes {
  std::vector<TensorIndex> src_idxs;
  std::vector<TensorIndex> dst_idxs;
  std::vector<TensorDim> src_dims;
  std::vector<TensorDim> dst_dims;
  std::set<size_t> axes;

  AggregationAxes(size_t ndims, const Value& in_axes, bool keepdims) : src_idxs(ndims), src_dims(ndims) {
    if (in_axes.is_none()) {
      for (size_t i = 0; i < ndims; i++) {
        axes.insert(i);
      }
    } else if (in_axes.is_tuple()) {
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

Value convolution(const Value& value) {
  auto args = value.as_tuple();
  if (args.size() != 8) {
    throw std::runtime_error("convolution expects 8 arguments");
  }

  auto I = args[0].as_tensor();  // input
  auto I_shape = I.shape();
  auto I_layout = args[6].as_str();
  if (I_shape.layout().size()) {
    I_layout = I_shape.layout();
  }
  if (I_layout.empty()) {
    I_layout = NHWC;
  }
  if (I_layout != NHWC && I_layout != NCHW) {
    throw std::runtime_error(str(boost::format("Unsupported layout for input of convolution: %1%") % I_layout));
  }
  IVLOG(2, "I_layout: " << I_layout);

  auto K = args[1].as_tensor();  // kernel/weights
  auto K_shape = K.shape();
  auto K_layout = args[7].as_str();
  if (K_shape.layout().size()) {
    K_layout = K_shape.layout();
  }
  if (K_layout.empty()) {
    K_layout = HWCK;
  }
  if (K_layout != HWCK && K_layout != KCHW) {
    throw std::runtime_error(str(boost::format("Unsupported layout for kernel of convolution: %1%") % K_layout));
  }
  IVLOG(2, "K_layout: " << K_layout);

  auto S = args[2].as_tuple();  // strides
  auto P = args[3].as_tuple();  // padding
  auto D = args[4].as_tuple();  // dilation
  // TODO: handle grouped convolutions
  // auto groups = args[5].as_int();

  auto ndims = S.size();

  TensorDim N, CI, CO;
  TensorIndex n, ci, co;

  std::vector<TensorDim> I_spatial_dims(ndims);
  std::vector<TensorDim> K_spatial_dims(ndims);
  std::vector<TensorDim> I_dims = {N};
  std::vector<TensorDim> K_dims;
  std::vector<TensorIndex> I_idxs = {n};
  std::vector<TensorIndex> O_idxs = {n};
  std::vector<TensorIndex> K_idxs;

  if (I_layout == NCHW) {
    I_idxs.emplace_back(ci);
    I_dims.emplace_back(CI);
    O_idxs.emplace_back(co);
  }
  if (K_layout == KCHW) {
    K_idxs.emplace_back(co);
    K_dims.emplace_back(CO);
    K_idxs.emplace_back(ci);
    K_dims.emplace_back(CI);
  }
  for (size_t i = 0; i < ndims; i++) {
    auto X = I_spatial_dims[i];
    auto K = K_spatial_dims[i];
    TensorIndex x;
    TensorIndex k;
    I_idxs.emplace_back(S[i].as_int() * x + D[i].as_int() * k - P[i].as_int());
    I_dims.emplace_back(X);
    K_idxs.push_back(k);
    K_dims.emplace_back(K);
    O_idxs.push_back(x);
  }
  if (I_layout == NHWC) {
    I_dims.emplace_back(CI);
    I_idxs.emplace_back(ci);
    O_idxs.emplace_back(co);
  }
  if (K_layout == HWCK) {
    K_idxs.emplace_back(ci);
    K_dims.emplace_back(CI);
    K_idxs.emplace_back(co);
    K_dims.emplace_back(CO);
  }

  I.bind_dims(I_dims);
  K.bind_dims(K_dims);

  std::vector<TensorDim> O_dims = {N};
  if (I_layout == NCHW) {
    O_dims.emplace_back(CO);
  }
  for (size_t i = 0; i < ndims; i++) {
    auto X = I_spatial_dims[i];
    auto K = K_spatial_dims[i];
    O_dims.emplace_back((X + 2 * P[i].as_int() - D[i].as_int() * (K - 1) - 1) / S[i].as_int() + 1);
  }
  if (I_layout == NHWC) {
    O_dims.emplace_back(CO);
  }

  auto O = TensorOutput(O_dims, I_layout);
  O(O_idxs) += I(I_idxs) * K(K_idxs);
  return Value{O};
}

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

Value square(const Value& value) {
  auto x = value.as_tensor();
  return Value(x * x);
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

void RegisterOps() {
  auto registry = OperationRegistry::Instance();
  registry->Register("convolution", convolution);
  registry->Register("mean", mean);
  registry->Register("square", square);
  registry->Register("sum", sum);
}

}  // namespace lib
}  // namespace op
}  // namespace plaidml
