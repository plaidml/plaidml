// RUN: pmlc-opt -convert-tile-to-linalg -canonicalize -cse %s | FileCheck %s

func @eltwise_and_f32(
  %arg0: tensor<3x3xf32>,
  %arg1: tensor<3x3xf32>
) -> tensor<3x3xi1> {
  %0 = tile.logical_and %arg1, %arg0 : (tensor<3x3xf32>, tensor<3x3xf32>) -> tensor<3x3xi1>
  return %0 : tensor<3x3xi1>
}

// CHECK-LABEL: func @eltwise_and_f32
// CHECK: constant
// CHECK: linalg.init_tensor
// CHECK: linalg.generic
// CHECK:   cmpf one
// CHECK:   cmpf one
// CHECK:   and
// CHECK:   linalg.yield

func @eltwise_or_si32(
  %arg0: tensor<3x3xsi32>,
  %arg1: tensor<3x3xsi32>
) -> tensor<3x3xi1> {
  %0 = tile.logical_or %arg1, %arg0 : (tensor<3x3xsi32>, tensor<3x3xsi32>) -> tensor<3x3xi1>
  return %0 : tensor<3x3xi1>
}

// CHECK-LABEL: func @eltwise_or_si32
// CHECK: constant
// CHECK: linalg.init_tensor
// CHECK: linalg.generic
// CHECK:   cmpi ne
// CHECK:   cmpi ne
// CHECK:   or
// CHECK:   linalg.yield

func @eltwise_xor_mixed(
  %arg0: tensor<3x3xf32>,
  %arg1: tensor<3x3xui64>
) -> tensor<3x3xi1> {
  %0 = tile.logical_xor %arg0, %arg1 : (tensor<3x3xf32>, tensor<3x3xui64>) -> tensor<3x3xi1>
  return %0 : tensor<3x3xi1>
}

// CHECK-LABEL: func @eltwise_xor_mixed
// CHECK: constant
// CHECK: linalg.init_tensor
// CHECK: linalg.generic
// CHECK:   cmpf one
// CHECK:   cmpi ne
// CHECK:   xor
// CHECK:   linalg.yield

func @eltwise_not_si32(
  %arg0: tensor<3x3xsi32>
) -> tensor<3x3xi1> {
  %0 = tile.logical_not %arg0 : (tensor<3x3xsi32>) -> tensor<3x3xi1>
  return %0 : tensor<3x3xi1>
}

// CHECK-LABEL: func @eltwise_not_si32
// CHECK: constant
// CHECK: linalg.init_tensor
// CHECK: linalg.generic
// CHECK:   cmpi eq
// CHECK:   linalg.yield

func @eltwise_not_f32(
  %arg0: tensor<3x3xf32>
) -> tensor<3x3xi1> {
  %0 = tile.logical_not %arg0 : (tensor<3x3xf32>) -> tensor<3x3xi1>
  return %0 : tensor<3x3xi1>
}

// CHECK-LABEL: func @eltwise_not_f32
// CHECK: constant
// CHECK: linalg.init_tensor
// CHECK: linalg.generic
// CHECK:   cmpf oeq
// CHECK:   linalg.yield
