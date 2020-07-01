// RUN: pmlc-opt -convert-tile-to-pxa -canonicalize -cse %s | FileCheck %s

// CHECK-LABEL: func @bit_not
func @bit_not(%arg0: tensor<10x20xsi32>) -> tensor<10x20xsi32> {
  // CHECK: %[[negOne:.*]] = constant -1 : i32
  // CHECK: affine.parallel
  // CHECK: affine.load
  // CHECK: subi %[[negOne]], %{{.*}}
  // CHECK: pxa.reduce assign
  %0 = "eltwise.bit_not"(%arg0) : (tensor<10x20xsi32>) -> tensor<10x20xsi32>
  return %0 : tensor<10x20xsi32>
}

// CHECK-LABEL: func @neg_i32
func @neg_i32(%arg0: tensor<10x20xsi32>) -> tensor<10x20xsi32> {
  // CHECK: %[[c0:.*]] = constant 0 : i32
  // CHECK: affine.parallel
  // CHECK: affine.load
  // CHECK: subi %[[c0]], %{{.*}}
  // CHECK: pxa.reduce assign
  %0 = "eltwise.neg"(%arg0) : (tensor<10x20xsi32>) -> tensor<10x20xsi32>
  return %0 : tensor<10x20xsi32>
}

// CHECK-LABEL: func @asin_f32
func @asin_f32(%arg0: tensor<8x9xf32>) -> tensor<8x9xf32> {
  // CHECK: affine.parallel
  // CHECK: affine.load
  // CHECK: stdx.asin({{.*}}) : (f32) -> f32
  // CHECK: pxa.reduce assign
  %0 = "eltwise.asin"(%arg0) : (tensor<8x9xf32>) -> tensor<8x9xf32>
  return %0 : tensor<8x9xf32>
}

