// RUN: pmlc-opt -convert-tile-to-pxa -canonicalize -cse %s | FileCheck %s

func @eltwise_and_f32(
  %arg0: tensor<3x3xf32>,
  %arg1: tensor<3x3xf32>
) -> tensor<3x3xi1> {
  %0 = "eltwise.logical_and"(%arg1, %arg0) : (
    tensor<3x3xf32>,
    tensor<3x3xf32>
  ) -> tensor<3x3xi1>
  return %0 : tensor<3x3xi1>
}

// CHECK-LABEL: func @eltwise_and_f32
// CHECK: affine.parallel
// CHECK: pxa.load
// CHECK: pxa.load
// CHECK: cmpf "one"
// CHECK: cmpf "one"
// CHECK: and
// CHECK: pxa.reduce assign

func @eltwise_or_si32(
  %arg0: tensor<3x3xsi32>,
  %arg1: tensor<3x3xsi32>
) -> tensor<3x3xi1> {
  %0 = "eltwise.logical_or"(%arg1, %arg0) : (
    tensor<3x3xsi32>,
    tensor<3x3xsi32>
  ) -> tensor<3x3xi1>
  return %0 : tensor<3x3xi1>
}

// CHECK-LABEL: func @eltwise_or_si32
// CHECK: affine.parallel
// CHECK: pxa.load
// CHECK: pxa.load
// CHECK: cmpi "ne"
// CHECK: cmpi "ne"
// CHECK: or
// CHECK: pxa.reduce assign

func @eltwise_xor_mixed(
  %arg0: tensor<3x3xf32>,
  %arg1: tensor<3x3xui64>
) -> tensor<3x3xi1> {
  %0 = "eltwise.logical_xor"(%arg0, %arg1) : (
    tensor<3x3xf32>,
    tensor<3x3xui64>
  ) -> tensor<3x3xi1>
  return %0 : tensor<3x3xi1>
}

// CHECK-LABEL: func @eltwise_xor_mixed
// CHECK: affine.parallel
// CHECK: pxa.load
// CHECK: pxa.load
// CHECK: cmpf "one"
// CHECK: cmpi "ne"
// CHECK: xor
// CHECK: pxa.reduce assign

func @eltwise_not_si32(
  %arg0: tensor<3x3xsi32>
) -> tensor<3x3xi1> {
  %0 = "eltwise.logical_not"(%arg0) : (
    tensor<3x3xsi32>
  ) -> tensor<3x3xi1>
  return %0 : tensor<3x3xi1>
}

// CHECK-LABEL: func @eltwise_not_si32
// CHECK: constant 0
// CHECK: affine.parallel
// CHECK: pxa.load
// CHECK: cmpi "eq"
// CHECK: pxa.reduce assign

func @eltwise_not_f32(
  %arg0: tensor<3x3xf32>
) -> tensor<3x3xi1> {
  %0 = "eltwise.logical_not"(%arg0) : (
    tensor<3x3xf32>
  ) -> tensor<3x3xi1>
  return %0 : tensor<3x3xi1>
}

// CHECK-LABEL: func @eltwise_not_f32
// CHECK: constant 0{{.*}} : f32
// CHECK: affine.parallel
// CHECK: pxa.load
// CHECK: cmpf "oeq"
// CHECK: pxa.reduce assign
