// RUN: pmlc-opt -convert-tile-to-pxa -canonicalize -cse %s | FileCheck %s

// CHECK-LABEL: func @bit_not
func @bit_not(%arg0: tensor<10x20xsi32>) -> tensor<10x20xsi32> {
  // CHECK: %[[negOne:.*]] = constant -1 : i32
  // CHECK: affine.parallel
  // CHECK: pxa.load
  // CHECK: subi %[[negOne]], %{{.*}}
  // CHECK: pxa.reduce assign
  %0 = "eltwise.bit_not"(%arg0) : (tensor<10x20xsi32>) -> tensor<10x20xsi32>
  return %0 : tensor<10x20xsi32>
}

// CHECK-LABEL: func @neg_i32
func @neg_i32(%arg0: tensor<10x20xsi32>) -> tensor<10x20xsi32> {
  // CHECK: %[[c0:.*]] = constant 0 : i32
  // CHECK: affine.parallel
  // CHECK: pxa.load
  // CHECK: subi %[[c0]], %{{.*}}
  // CHECK: pxa.reduce assign
  %0 = "eltwise.neg"(%arg0) : (tensor<10x20xsi32>) -> tensor<10x20xsi32>
  return %0 : tensor<10x20xsi32>
}

// CHECK-LABEL: func @acos_f32
func @acos_f32(%arg0: tensor<8x9xf32>) -> tensor<8x9xf32> {
  // CHECK: affine.parallel
  // CHECK: pxa.load
  // CHECK: stdx.acos({{.*}}) : (f32) -> f32
  // CHECK: pxa.reduce assign
  %0 = "eltwise.acos"(%arg0) : (tensor<8x9xf32>) -> tensor<8x9xf32>
  return %0 : tensor<8x9xf32>
}

// CHECK-LABEL: func @asin_f32
func @asin_f32(%arg0: tensor<8x9xf32>) -> tensor<8x9xf32> {
  // CHECK: affine.parallel
  // CHECK: pxa.load
  // CHECK: stdx.asin({{.*}}) : (f32) -> f32
  // CHECK: pxa.reduce assign
  %0 = "eltwise.asin"(%arg0) : (tensor<8x9xf32>) -> tensor<8x9xf32>
  return %0 : tensor<8x9xf32>
}

// CHECK-LABEL: func @atan_f32
func @atan_f32(%arg0: tensor<8x9xf32>) -> tensor<8x9xf32> {
  // CHECK: affine.parallel
  // CHECK: pxa.load
  // CHECK: stdx.atan({{.*}}) : (f32) -> f32
  // CHECK: pxa.reduce assign
  %0 = "eltwise.atan"(%arg0) : (tensor<8x9xf32>) -> tensor<8x9xf32>
  return %0 : tensor<8x9xf32>
}

// CHECK-LABEL: func @cosh_f32
func @cosh_f32(%arg0: tensor<8x9xf32>) -> tensor<8x9xf32> {
  // CHECK: affine.parallel
  // CHECK: pxa.load
  // CHECK: stdx.cosh({{.*}}) : (f32) -> f32
  // CHECK: pxa.reduce assign
  %0 = "eltwise.cosh"(%arg0) : (tensor<8x9xf32>) -> tensor<8x9xf32>
  return %0 : tensor<8x9xf32>
}

// CHECK-LABEL: func @erf_f32
func @erf_f32(%arg0: tensor<8x9xf32>) -> tensor<8x9xf32> {
  // CHECK: affine.parallel
  // CHECK: pxa.load
  // CHECK: stdx.erf({{.*}}) : (f32) -> f32
  // CHECK: pxa.reduce assign
  %0 = "eltwise.erf"(%arg0) : (tensor<8x9xf32>) -> tensor<8x9xf32>
  return %0 : tensor<8x9xf32>
}

// CHECK-LABEL: func @floor_f32
func @floor_f32(%arg0: tensor<8x9xf32>) -> tensor<8x9xf32> {
  // CHECK: affine.parallel
  // CHECK: pxa.load
  // CHECK: stdx.floor({{.*}}) : (f32) -> f32
  // CHECK: pxa.reduce assign
  %0 = "eltwise.floor"(%arg0) : (tensor<8x9xf32>) -> tensor<8x9xf32>
  return %0 : tensor<8x9xf32>
}

// CHECK-LABEL: func @pow_f32
func @pow_f32(%arg0: tensor<8x9xf32>, %arg1: tensor<8x9xf32>) -> tensor<8x9xf32> {
  // CHECK: affine.parallel
  // CHECK: pxa.load
  // CHECK: stdx.pow({{.*}}, {{.*}}) : (f32, f32) -> f32
  // CHECK: pxa.reduce assign
  %0 = "eltwise.pow"(%arg0, %arg1) : (tensor<8x9xf32>, tensor<8x9xf32>) -> tensor<8x9xf32>
  return %0 : tensor<8x9xf32>
}

// CHECK-LABEL: func @pow_f32_to_si32
func @pow_f32_to_si32(%arg0: tensor<8x9xf32>, %arg1: tensor<8x9xsi32>) -> tensor<8x9xf32> {
  // CHECK: affine.parallel
  // CHECK: pxa.load
  // CHECK: stdx.pow({{.*}}, {{.*}}) : (f32, f32) -> f32
  // CHECK: pxa.reduce assign
  %0 = "eltwise.pow"(%arg0, %arg1) : (tensor<8x9xf32>, tensor<8x9xsi32>) -> tensor<8x9xf32>
  return %0 : tensor<8x9xf32>
}

// CHECK-LABEL: func @round_f32
func @round_f32(%arg0: tensor<8x9xf32>) -> tensor<8x9xf32> {
  // CHECK: affine.parallel
  // CHECK: pxa.load
  // CHECK: stdx.round({{.*}}) : (f32) -> f32
  // CHECK: pxa.reduce assign
  %0 = "eltwise.round"(%arg0) : (tensor<8x9xf32>) -> tensor<8x9xf32>
  return %0 : tensor<8x9xf32>
}

// CHECK-LABEL: func @sinh_f32
func @sinh_f32(%arg0: tensor<8x9xf32>) -> tensor<8x9xf32> {
  // CHECK: affine.parallel
  // CHECK: pxa.load
  // CHECK: stdx.sinh({{.*}}) : (f32) -> f32
  // CHECK: pxa.reduce assign
  %0 = "eltwise.sinh"(%arg0) : (tensor<8x9xf32>) -> tensor<8x9xf32>
  return %0 : tensor<8x9xf32>
}

// CHECK-LABEL: func @tan_f32
func @tan_f32(%arg0: tensor<8x9xf32>) -> tensor<8x9xf32> {
  // CHECK: affine.parallel
  // CHECK: pxa.load
  // CHECK: stdx.tan({{.*}}) : (f32) -> f32
  // CHECK: pxa.reduce assign
  %0 = "eltwise.tan"(%arg0) : (tensor<8x9xf32>) -> tensor<8x9xf32>
  return %0 : tensor<8x9xf32>
}

// CHECK-LABEL: func @sin
func @sin(%arg0: tensor<8x9xf32>) -> tensor<8x9xf32> {
  // CHECK: affine.parallel
  // CHECK: pxa.load
  // CHECK: sin{{.*}} : f32
  // CHECK: pxa.reduce assign
  %0 = "eltwise.sin"(%arg0) : (tensor<8x9xf32>) -> tensor<8x9xf32>
  return %0 : tensor<8x9xf32>
}
