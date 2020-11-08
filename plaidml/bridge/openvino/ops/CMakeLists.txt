set(_SRCS
  abs.cpp
  acos.cpp
  add.cpp
  asin.cpp
  atan.cpp
  avg_pool.cpp
  batch_norm_inference.cpp
  batch_to_space.cpp
  binary_convolution.cpp
  broadcast.cpp
  bucketize.cpp
  ceiling.cpp
  clamp.cpp
  concat.cpp
  convert_like.cpp
  convert.cpp
  convolution_backprop_data.cpp
  convolution.cpp
  cos.cpp
  cosh.cpp
  depth_to_space.cpp
  divide.cpp
  elu.cpp
  equal.cpp
  erf.cpp
  exp.cpp
  fake_quantize.cpp
  floor_mod.cpp
  floor.cpp
  greater_equal.cpp
  greater.cpp
  grn.cpp
  group_convolution_backprop_data.cpp
  group_convolution.cpp
  hard_sigmoid.cpp
  less_equal.cpp
  less.cpp
  log.cpp
  logical_and.cpp
  logical_not.cpp
  logical_or.cpp
  logical_xor.cpp
  lrn.cpp
  mat_mul.cpp
  max_pool.cpp
  maximum.cpp
  minimum.cpp
  mod.cpp
  multiply.cpp
  mvn.cpp
  negative.cpp
  normalize_l2.cpp
  not_equal.cpp
  pad.cpp
  power.cpp
  prelu.cpp
  reduce_logical_and.cpp
  reduce_logical_or.cpp
  reduce_max.cpp
  reduce_mean.cpp
  reduce_min.cpp
  reduce_prod.cpp
  reduce_sum.cpp
  relu.cpp
  reshape.cpp
  reverse.cpp
  select.cpp
  selu.cpp
  sigmoid.cpp
  sign.cpp
  sin.cpp
  sinh.cpp
  softmax.cpp
  space_to_depth.cpp
  split.cpp
  sqrt.cpp
  squared_difference.cpp
  squeeze.cpp
  strided_slice.cpp
  subtract.cpp
  swish.cpp
  tan.cpp
  tanh.cpp
  tile.cpp
  transpose.cpp
  unsqueeze.cpp
  variadic_split.cpp
)
set(_TODO_FAKE_SRCS add.cpp)
# set(_HDRS plaidml_program_builder.hpp)  # TODO Clean

pml_cc_library(
  NAME ops
  TYPE STATIC
  # HDRS ${_HDRS}  # TODO Clean
  SRCS ${_TODO_FAKE_SRCS}
  # TODO: Confirm & clean DEPS
  DEPS
    # plaidml::bridge::openvino
    # plaidml::core
    # IE::inference_engine
    # pmlc::ast
    # pmlc::util
    # LLVMSupport
)
