// RUN: pmlc-opt -diag-conv2d-finder --split-input-file %s 2>&1 | FileCheck %s

#conv2d_1x1_0 = affine_map<(d0, d1, d2, d3, d4, d5, d6) -> (d0, d1, d2, d3)>
#conv2d_1x1_1 = affine_map<(d0, d1, d2, d3, d4, d5, d6) -> (d0, d1 + d4, d2 + d5, d6)>
#conv2d_1x1_2 = affine_map<(d0, d1, d2, d3, d4, d5, d6) -> (d4, d5, d6, d3)>

func @conv2d_1x1_nhwc(%arg0: tensor<1x224x224x3xf32>, %arg1: tensor<1x1x3x32xf32>) -> tensor<1x224x224x32xf32> {
  %cst = tile.constant(0.000000e+00 : f64) : tensor<f32>
  // CHECK: You say you want a convolution.
  // CHECK: paddings = 0 0
  // CHECK: strides = 1 1
  // CHECK: dilations = 1 1
  %0 = tile.contract add, mul, %cst, %arg0, %arg1 {sink = #conv2d_1x1_0, srcs = [#conv2d_1x1_1, #conv2d_1x1_2]} : tensor<f32>, tensor<1x224x224x3xf32>, tensor<1x1x3x32xf32> -> tensor<1x224x224x32xf32>
  return %0 : tensor<1x224x224x32xf32>
}

#conv2d_3x3_0 = affine_map<(d0, d1, d2, d3, d4, d5, d6) -> (d0, d1, d2, d3)>
#conv2d_3x3_1 = affine_map<(d0, d1, d2, d3, d4, d5, d6) -> (d0, d1 + d4 - 1, d2 + d5 - 1, d6)>
#conv2d_3x3_2 = affine_map<(d0, d1, d2, d3, d4, d5, d6) -> (d4, d5, d6, d3)>

func @conv2d_3x3_nhwc(%arg0: tensor<1x224x224x3xf32>, %arg1: tensor<3x3x3x32xf32>) -> tensor<1x224x224x32xf32> {
  %cst = tile.constant(0.000000e+00 : f64) : tensor<f32>
  // CHECK: You say you want a convolution.
  // CHECK: paddings = -1 -1
  // CHECK: strides = 1 1
  // CHECK: dilations = 1 1
  %0 = tile.contract add, mul, %cst, %arg0, %arg1 {sink = #conv2d_3x3_0, srcs = [#conv2d_3x3_1, #conv2d_3x3_2]} : tensor<f32>, tensor<1x224x224x3xf32>, tensor<3x3x3x32xf32> -> tensor<1x224x224x32xf32>
  return %0 : tensor<1x224x224x32xf32>
}
