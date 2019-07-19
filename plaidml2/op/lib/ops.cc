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
  std::vector<TensorIndex> reduce_idxs;
  std::vector<TensorDim> src_dims;
  std::vector<TensorDim> dst_dims;
  std::vector<TensorDim> reduce_dims;
  std::set<size_t> axes;

  AggregationAxes(size_t ndims, const Value& in_axes, bool keepdims) : src_idxs(ndims), src_dims(ndims) {
    if (in_axes.is_none()) {
      for (size_t i = 0; i < ndims; i++) {
        axes.insert(i);
      }
    } else if (in_axes.is_tuple()) {
      for (const auto& axis : in_axes.as_tuple()) {
        auto int_axis = axis.as_int();
        if (int_axis < 0) {
          int_axis = ndims + int_axis;
        }
        if (int_axis < 0 || ndims < static_cast<size_t>(int_axis)) {
          throw std::out_of_range(str(boost::format("axis out of range: %1%") % int_axis));
        }
        axes.insert(int_axis);
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
    for (auto axis : axes) {
      reduce_idxs.push_back(src_idxs[axis]);
      reduce_dims.push_back(src_dims[axis]);
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

enum class AutopadMode : char {
  NONE = '-',
  NOTSET = NONE,
  EXPLICIT = NONE,
  SAME_LOWER = 'L',
  SAME_UPPER = 'U',
  VALID = 'V'
};

enum class PoolMode : char { AVG = 'A', MAX = '>', MIN = '<', SUM = '+' };

enum class TensorLayout { NXC, NCX, KCX, XCK };

namespace {
// TODO: I haven't decided whether to make these helper functions visible to the outside world

AutopadMode autopad_mode_from_str(const std::string& s) {
  if (s == "none") {
    return AutopadMode::NONE;
  }
  if (s == "notset") {
    return AutopadMode::NOTSET;
  }
  if (s == "explicit") {
    return AutopadMode::EXPLICIT;
  }
  if (s == "same_lower") {
    return AutopadMode::SAME_LOWER;
  }
  if (s == "same_upper") {
    return AutopadMode::SAME_UPPER;
  }
  if (s == "valid") {
    return AutopadMode::VALID;
  }
  throw std::runtime_error(str(boost::format("Unable to parse string '%1%' as an autopadding mode") % s));
}

std::string to_string(AutopadMode m) {
  if (m == AutopadMode::NONE) {
    return "none";
  }
  if (m == AutopadMode::SAME_LOWER) {
    return "same_lower";
  }
  if (m == AutopadMode::SAME_UPPER) {
    return "same_upper";
  }
  if (m == AutopadMode::VALID) {
    return "valid";
  }
  throw std::runtime_error("Unable to autopadding mode to string due to unrecognized mode");
}

PoolMode pool_mode_from_str(const std::string& s) {
  if (s == "avg" || s == "average") {
    return PoolMode::AVG;
  }
  if (s == "max") {
    return PoolMode::MAX;
  }
  if (s == "min") {
    return PoolMode::MIN;
  }
  if (s == "sum") {
    return PoolMode::SUM;
  }
  throw std::runtime_error(str(boost::format("Unable to parse string '%1%' as a pooling mode") % s));
}

TensorLayout tensor_layout_from_str(const std::string& s) {
  if (s == "nxc" || s == "nwc" || s == "nhwc" || s == "ndhwc") {
    return TensorLayout::NXC;
  }
  if (s == "ncx" || s == "ncw" || s == "nchw" || s == "ncdhw") {
    return TensorLayout::NCX;
  }
  if (s == "kcx" || s == "kcw" || s == "kchw" || s == "kcdhw") {
    return TensorLayout::KCX;
  }
  if (s == "xck" || s == "wck" || s == "hwck" || s == "dhwck") {
    return TensorLayout::XCK;
  }
  throw std::runtime_error(str(boost::format("Unable to parse string '%1%' as a tensor layout") % s));
}

bool is_input_layout(TensorLayout layout) {  //
  return (layout == TensorLayout::NCX || layout == TensorLayout::NXC);
}

// TODO: Uncomment once we actually have a use for this
// bool is_kernel_layout(TensorLayout layout) {
//   return (layout == TensorLayout::KCX || layout == TensorLayout::XCK);
// }

std::pair<TensorDim, TensorDim> compute_padding_and_output_size(const TensorDim& input_size,
                                                                const TensorDim& effective_filter_size, int64_t stride,
                                                                AutopadMode autopad_mode, int64_t pad_lo,
                                                                int64_t pad_hi, bool use_ceil_for_output_shape) {
  // effective_filter_size is the filter size after dilation is accounted for. So a 4x3 filter dilated by (3, 2) has
  // effective_filter_sizes of 11 and 5 for its two spatial dimensions

  int64_t ceil_term =
      use_ceil_for_output_shape ? stride - 1 : 0;  // TODO: Will need to confirm that this is the intended behavior
  if (autopad_mode == AutopadMode::NONE) {
    TensorDim pad_before(pad_lo);
    TensorDim output_size((input_size + pad_lo + pad_hi - effective_filter_size + stride + ceil_term) / stride);
    return std::pair<TensorDim, TensorDim>(pad_before, output_size);
  }
  if (autopad_mode == AutopadMode::VALID) {
    TensorDim pad_before(0);
    TensorDim output_size((input_size - effective_filter_size + stride + ceil_term) / stride);
    return std::pair<TensorDim, TensorDim>(pad_before, output_size);
  }
  if (autopad_mode == AutopadMode::SAME_LOWER || autopad_mode == AutopadMode::SAME_UPPER) {
    TensorDim output_size((input_size + stride - 1 + ceil_term) / stride);
    int64_t upper_term = (autopad_mode == AutopadMode::SAME_UPPER) ? 1 : 0;
    // TensorDim pad_before((max(0, (output_size - 1) * stride + effective_filter_size - input_size) + upper_term) / 2);
    // TODO: Switch to above once max(TensorDim, TensorDim) is working
    TensorDim pad_before(((output_size - 1) * stride + effective_filter_size - input_size + upper_term) / 2);
    return std::pair<TensorDim, TensorDim>(pad_before, output_size);
  }
  throw std::runtime_error(str(boost::format("Unexpected autopadding mode: %1%") % to_string(autopad_mode)));
}

std::vector<int64_t>* extend_manual_padding(std::vector<int64_t>* pads, size_t rank) {
  if (pads->size() > 2 * rank) {
    throw std::runtime_error(str(
        boost::format(
            "Inconsistent spatial rank: operation with %1% spatial dimensions had %2% manual padding values given") %
        rank % pads->size()));
  }
  while (pads->size() < 2 * rank) {
    pads->push_back(0);
  }
  return pads;
}

}  // namespace

Value argmax(const Value& value) {
  IVLOG(1, "argmax");
  auto args = value.as_tuple();
  if (args.size() != 2) {
    throw std::runtime_error("argmax expects 2 arguments");
  }
  auto I = args[0].as_tensor();
  auto I_shape = I.shape();
  auto axes = args[1];
  AggregationAxes agg(I_shape.ndims(), axes, false);
  I.bind_dims(agg.src_dims);
  auto M = TensorOutput(agg.dst_dims);
  M(agg.dst_idxs) >= I(agg.src_idxs);
  Tensor One(1);
  auto T = TensorOutput(agg.reduce_dims);
  T(agg.reduce_idxs) = One();
  auto IX = index(T, 0);
  auto AM = TensorOutput(agg.dst_dims);
  AM(agg.dst_idxs) >= cond(I(agg.src_idxs), M(agg.dst_idxs), IX(agg.reduce_idxs));
  auto O = as_uint(AM, 32);
  return Value{O};
}

Value convolution(const Value& value) {
  IVLOG(1, "convolution");
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

Value dot(const Value& value) {
  IVLOG(1, "dot");
  auto args = value.as_tuple();
  if (args.size() != 2) {
    throw std::runtime_error("dot expects 2 arguments");
  }
  auto X = args[0].as_tensor();
  auto X_shape = X.shape();
  auto Y = args[1].as_tensor();
  auto Y_shape = Y.shape();
  if (X_shape.dtype() != Y_shape.dtype()) {
    throw std::runtime_error(str(boost::format("Invalid dtype in dot: X.dtype = '%1%', Y.dtype = '%2%'") %
                                 X_shape.dtype() % Y_shape.dtype()));
  }
  if (X_shape.ndims() == 1 && Y_shape.ndims() == 1) {
    TensorDim I;
    TensorIndex i;
    X.bind_dims(I);
    Y.bind_dims(I);
    auto O = TensorOutput(I);
    O(i) += X(i) * Y(i);
    return Value{O};
  }
  if (1 <= X_shape.ndims() && 2 <= Y_shape.ndims()) {
    std::vector<TensorDim> X_dims(X_shape.ndims());
    std::vector<TensorDim> Y_dims(Y_shape.ndims());
    TensorIndex z;
    std::vector<TensorIndex> X_idxs(X_shape.ndims());
    std::vector<TensorIndex> Y_idxs(Y_shape.ndims());
    X_idxs[X_shape.ndims() - 1] = z;
    Y_idxs[Y_shape.ndims() - 2] = z;
    X.bind_dims(X_dims);
    Y.bind_dims(Y_dims);
    std::vector<TensorDim> O_dims;
    std::vector<TensorIndex> O_idxs;
    for (size_t i = 0; i < X_shape.ndims() - 1; i++) {
      O_dims.push_back(X_dims[i]);
      O_idxs.push_back(X_idxs[i]);
    }
    for (size_t i = 0; i < Y_shape.ndims() - 2; i++) {
      O_dims.push_back(Y_dims[i]);
      O_idxs.push_back(Y_idxs[i]);
    }
    O_dims.push_back(Y_dims[Y_shape.ndims() - 1]);
    O_idxs.push_back(Y_idxs[Y_shape.ndims() - 1]);
    auto O = TensorOutput(O_dims);
    O(O_idxs) += X(X_idxs) * Y(Y_idxs);
    return Value{O};
  }
  throw std::runtime_error(str(boost::format("Unsupported dims for dot operation: X.dims = %1%, Y.dims = %2%") %
                               X_shape.ndims() % Y_shape.ndims()));
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
    denom = denom * agg.src_dims.at(axis);
  }
  return Value{SO / denom};
}

Value pool(const Value& value) {
  // The parameters of pool:
  //    0. Input Tensor
  //    1. Pool Mode (avg/max)
  //    2. Pool Size
  //    3. Strides
  //    4. Autopad Mode (explicit, same_lower, same_upper, valid, [maybe full?])
  //    5. Manual Padding
  //    6. Layout (i.e. Channel Order) (minimally NXC v NCX)
  //    7. Include Padding in Avg Computation (bool)
  //    8. Ceil Mode (i.e. as in ONNX)
  // n.b. We determine the number of spatial dimensions from the Pool Size and confirm it is consistent with other
  // parameters that imply a spatial dimension size, specifically strides. We do also check this against the input
  // tensor shape and the manual padding, but these are less strict: manual padding may omit some padding values
  // (which are then assumed to be 0), and the input tensor shape may have multiple channel dimensions (i.e. for
  // cases like tensors going into or coming out of grouped convolutions).

  // Read arguments
  auto args = value.as_tuple();
  if (args.size() != 9) {
    throw std::runtime_error(str(boost::format("PlaidML pool op expects 9 arguments (received %1%)") % args.size()));
  }
  auto I = args[0].as_tensor();
  auto pool_mode = pool_mode_from_str(args[1].as_str());
  auto pool_size = args[2].as_int_tuple();
  auto strides = args[3].as_int_tuple();
  auto autopad_mode = autopad_mode_from_str(args[4].as_str());
  auto manual_padding = args[5].as_int_tuple();
  auto input_layout = tensor_layout_from_str(args[6].as_str());
  auto include_padding_in_avg = args[7].as_bool();
  auto use_ceil_for_output_shape = args[8].as_bool();

  // Initialize useful values
  auto spatial_rank = pool_size.size();
  auto I_shape = I.shape();
  auto I_channel_dims = I_shape.ndims() - spatial_rank - 1;

  // Verify inputs are consistent
  if (manual_padding.size() && autopad_mode != AutopadMode::NONE) {
    throw std::runtime_error("Autopadding and manual padding both requested for single pool operation");
  }
  if (strides.size() != spatial_rank) {
    throw std::runtime_error(
        str(boost::format("Inconsistent spatial rank in pool op (received %1%D pool_size and %2%D strides)") %
            spatial_rank % strides.size()));
  }
  if (I_channel_dims < 1) {
    throw std::runtime_error(
        str(boost::format("Inconsistent spatial rank in pool op (pool_size has %1% spatial dimensions but input tensor "
                          "has %2% dimensions, and thus at most %3% spatial dims)") %
            spatial_rank % I.shape().ndims() % (I.shape().ndims() - 2)));
  }
  if (!is_input_layout(input_layout)) {
    throw std::runtime_error("Tensor layout requested in pool op does not apply to pooling");
  }

  extend_manual_padding(&manual_padding, spatial_rank);

  TensorDim N, C;
  TensorIndex n, c;
  std::vector<TensorDim> X(spatial_rank);
  std::vector<TensorIndex> x(spatial_rank);
  std::vector<TensorIndex> k(spatial_rank);  // within-pool spatial indexes

  std::vector<TensorDim> pad_before;
  std::vector<TensorDim> I_dims = {N};
  std::vector<TensorIndex> I_idxs = {n};
  std::vector<TensorDim> O_dims = {N};
  std::vector<TensorIndex> O_idxs = {n};
  if (input_layout == TensorLayout::NCX) {
    I_dims.push_back(C);
  }
  for (size_t i = 0; i < spatial_rank; ++i) {
    I_dims.push_back(X[i]);
  }
  if (input_layout == TensorLayout::NXC) {
    I_dims.push_back(C);
  }
  I.bind_dims(I_dims);
  if (input_layout == TensorLayout::NCX) {
    I_idxs.push_back(c);
    O_dims.push_back(C);
    O_idxs.push_back(c);
  }
  for (size_t i = 0; i < spatial_rank; ++i) {
    O_idxs.push_back(x[i]);
    TensorDim local_pad_before;
    TensorDim local_output_size;
    TensorIndex local_index;
    std::tie(local_pad_before, local_output_size) =
        compute_padding_and_output_size(X[i], TensorDim(pool_size[i]), strides[i], autopad_mode, manual_padding[2 * i],
                                        manual_padding[2 * i + 1], use_ceil_for_output_shape);
    pad_before.emplace_back(local_pad_before);
    local_index = strides[i] * x[i] + k[i] - pad_before[i];
    O_dims.emplace_back(local_output_size);
    I_idxs.emplace_back(local_index);
    // Add constraints: the format is weird but this does actually apply the constraints
    if (k[i] < pool_size[i]) {
      // nolint(whitespace/empty_if_body)
    }
  }
  if (input_layout == TensorLayout::NXC) {
    I_idxs.push_back(c);
    O_dims.push_back(C);
    O_idxs.push_back(c);
  }
  auto O = TensorOutput(O_dims);
  if (pool_mode == PoolMode::MAX) {
    O(O_idxs) >= I(I_idxs);
    return Value{O};
  } else if (pool_mode == PoolMode::MIN) {
    O(O_idxs) <= I(I_idxs);
    return Value{O};
  } else if (pool_mode == PoolMode::SUM) {
    O(O_idxs) += I(I_idxs);
    return Value{O};
  } else if (pool_mode == PoolMode::AVG) {
    O(O_idxs) += I(I_idxs);
    if (include_padding_in_avg) {
      int64_t total_pool_size = 1;
      for (const auto& sz : pool_size) {
        total_pool_size *= sz;
      }
      return Value{O / total_pool_size};
    } else {
      auto One = Tensor{1};
      auto Ones = TensorOutput(I_dims);
      auto Count = TensorOutput(O_dims);
      // Note: O_idxs is used in both cases b/c both need indexes of the form x0, x1, ...
      // However, they do not represent the same index values (and notably do not interate
      // over the same size of dimensions as I_dims != O_dims)
      Ones(O_idxs) = One(std::vector<TensorIndex>());
      Count(O_idxs) += Ones(I_idxs);
      return Value{O / Count};
    }
  } else {
    throw std::runtime_error("Unrecognized pool_mode in pool op");
  }
}

Value relu(const Value& value) {
  IVLOG(1, "relu");
  auto args = value.as_tuple();
  if (args.size() != 4) {
    throw std::runtime_error("relu expects 4 arguments");
  }
  auto I = args[0].as_tensor();
  auto alpha = args[1];
  auto max_value = args[2];
  auto threshold = args[3].as_float();
  Tensor A;
  if (alpha.is_none()) {
    A = Tensor(0.0);
  } else {
    A = alpha.as_tensor();
  }
  auto O = select(I < threshold, A * (I - threshold), I);
  if (!max_value.is_none()) {
    auto M = max_value.as_tensor();
    O = select(O < M, O, M);
  }
  return Value{O};
}

Value softmax(const Value& value) {
  IVLOG(1, "softmax");
  auto args = value.as_tuple();
  if (args.size() != 2) {
    throw std::runtime_error("softmax expects 2 arguments");
  }
  auto X = args[0].as_tensor();
  if (X.shape().ndims() == 2) {
    TensorDim I, J;
    TensorIndex i, j;
    X.bind_dims(I, J);
    auto M = TensorOutput(I, 1);
    M(i, 0) >= X(i, j);
    auto E = exp(X - M);
    auto N = TensorOutput(I, 1);
    N(i, 0) += E(i, j);
    return Value{E / N};
  }
  throw std::runtime_error("softmax only works on 2 dimensions at this time.");
}

Value square(const Value& value) {
  IVLOG(1, "square");
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
  registry->Register("argmax", argmax);
  registry->Register("convolution", convolution);
  registry->Register("dot", dot);
  registry->Register("mean", mean);
  registry->Register("pool", pool);
  registry->Register("relu", relu);
  registry->Register("softmax", softmax);
  registry->Register("square", square);
  registry->Register("sum", sum);
}

}  // namespace lib
}  // namespace op
}  // namespace plaidml
