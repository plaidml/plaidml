// RUN: pmlc-opt -convert-tile-to-linalg -canonicalize -cse %s | FileCheck %s

// CHECK-LABEL: func.func @bit_not
func.func @bit_not(%arg0: tensor<10x20xsi32>) -> tensor<10x20xsi32> {
  // CHECK: %[[negOne:.*]] = arith.constant -1 : i32
  // CHECK: linalg.init_tensor
  // CHECK: linalg.generic
  // CHECK:   subi %[[negOne]], %{{.*}}
  // CHECK:   linalg.yield
  %0 = tile.bit_not %arg0 : (tensor<10x20xsi32>) -> tensor<10x20xsi32>
  return %0 : tensor<10x20xsi32>
}

// CHECK-LABEL: func.func @neg_i32
func.func @neg_i32(%arg0: tensor<10x20xsi32>) -> tensor<10x20xsi32> {
  // CHECK: %[[c0:.*]] = arith.constant 0 : i32
  // CHECK: linalg.init_tensor
  // CHECK: linalg.generic
  // CHECK:   subi %[[c0]], %{{.*}}
  // CHECK:   linalg.yield
  %0 = tile.neg %arg0 : (tensor<10x20xsi32>) -> tensor<10x20xsi32>
  return %0 : tensor<10x20xsi32>
}

// CHECK-LABEL: func.func @acos_f32
func.func @acos_f32(%arg0: tensor<8x9xf32>) -> tensor<8x9xf32> {
  // CHECK: linalg.init_tensor
  // CHECK: linalg.generic
  // CHECK:   stdx.acos({{.*}}) : (f32) -> f32
  // CHECK:   linalg.yield
  %0 = tile.acos %arg0 : (tensor<8x9xf32>) -> tensor<8x9xf32>
  return %0 : tensor<8x9xf32>
}

// CHECK-LABEL: func.func @asin_f32
func.func @asin_f32(%arg0: tensor<8x9xf32>) -> tensor<8x9xf32> {
  // CHECK: linalg.init_tensor
  // CHECK: linalg.generic
  // CHECK:   stdx.asin({{.*}}) : (f32) -> f32
  // CHECK:   linalg.yield
  %0 = tile.asin %arg0 : (tensor<8x9xf32>) -> tensor<8x9xf32>
  return %0 : tensor<8x9xf32>
}

// CHECK-LABEL: func.func @atan_f32
func.func @atan_f32(%arg0: tensor<8x9xf32>) -> tensor<8x9xf32> {
  // CHECK: linalg.init_tensor
  // CHECK: linalg.generic
  // CHECK:   stdx.atan({{.*}}) : (f32) -> f32
  // CHECK:   linalg.yield
  %0 = tile.atan %arg0 : (tensor<8x9xf32>) -> tensor<8x9xf32>
  return %0 : tensor<8x9xf32>
}

// CHECK-LABEL: func.func @cosh_f32
func.func @cosh_f32(%arg0: tensor<8x9xf32>) -> tensor<8x9xf32> {
  // CHECK: linalg.init_tensor
  // CHECK: linalg.generic
  // CHECK:   stdx.cosh({{.*}}) : (f32) -> f32
  // CHECK:   linalg.yield
  %0 = tile.cosh %arg0 : (tensor<8x9xf32>) -> tensor<8x9xf32>
  return %0 : tensor<8x9xf32>
}

// CHECK-LABEL: func.func @erf_f32
func.func @erf_f32(%arg0: tensor<8x9xf32>) -> tensor<8x9xf32> {
  // CHECK: linalg.init_tensor
  // CHECK: linalg.generic
  // CHECK:   stdx.erf({{.*}}) : (f32) -> f32
  // CHECK:   linalg.yield
  %0 = tile.erf %arg0 : (tensor<8x9xf32>) -> tensor<8x9xf32>
  return %0 : tensor<8x9xf32>
}

// CHECK-LABEL: func.func @floor_f32
func.func @floor_f32(%arg0: tensor<8x9xf32>) -> tensor<8x9xf32> {
  // CHECK: linalg.init_tensor
  // CHECK: linalg.generic
  // CHECK:   stdx.floor({{.*}}) : (f32) -> f32
  // CHECK:   linalg.yield
  %0 = tile.floor %arg0 : (tensor<8x9xf32>) -> tensor<8x9xf32>
  return %0 : tensor<8x9xf32>
}

// CHECK-LABEL: func.func @pow_f32
func.func @pow_f32(%arg0: tensor<8x9xf32>, %arg1: tensor<8x9xf32>) -> tensor<8x9xf32> {
  // CHECK: linalg.init_tensor
  // CHECK: linalg.generic
  // CHECK:   stdx.pow({{.*}}, {{.*}}) : (f32, f32) -> f32
  // CHECK:   linalg.yield
  %0 = tile.pow %arg0, %arg1 : (tensor<8x9xf32>, tensor<8x9xf32>) -> tensor<8x9xf32>
  return %0 : tensor<8x9xf32>
}

// CHECK-LABEL: func.func @pow_f32_to_si32
func.func @pow_f32_to_si32(%arg0: tensor<8x9xf32>, %arg1: tensor<8x9xsi32>) -> tensor<8x9xf32> {
  // CHECK: linalg.init_tensor
  // CHECK: linalg.generic
  // CHECK:   sitofp
  // CHECK:   stdx.pow({{.*}}, {{.*}}) : (f32, f32) -> f32
  // CHECK:   linalg.yield
  %0 = tile.pow %arg0, %arg1 : (tensor<8x9xf32>, tensor<8x9xsi32>) -> tensor<8x9xf32>
  return %0 : tensor<8x9xf32>
}

// CHECK-LABEL: func.func @round_f32
func.func @round_f32(%arg0: tensor<8x9xf32>) -> tensor<8x9xf32> {
  // CHECK: linalg.init_tensor
  // CHECK: linalg.generic
  // CHECK:   stdx.round({{.*}}) : (f32) -> f32
  // CHECK:   linalg.yield
  %0 = tile.round %arg0 : (tensor<8x9xf32>) -> tensor<8x9xf32>
  return %0 : tensor<8x9xf32>
}

// CHECK-LABEL: func.func @sinh_f32
func.func @sinh_f32(%arg0: tensor<8x9xf32>) -> tensor<8x9xf32> {
  // CHECK: linalg.init_tensor
  // CHECK: linalg.generic
  // CHECK:   stdx.sinh({{.*}}) : (f32) -> f32
  // CHECK:   linalg.yield
  %0 = tile.sinh %arg0 : (tensor<8x9xf32>) -> tensor<8x9xf32>
  return %0 : tensor<8x9xf32>
}

// CHECK-LABEL: func.func @tan_f32
func.func @tan_f32(%arg0: tensor<8x9xf32>) -> tensor<8x9xf32> {
  // CHECK: linalg.init_tensor
  // CHECK: linalg.generic
  // CHECK:   stdx.tan({{.*}}) : (f32) -> f32
  // CHECK:   linalg.yield
  %0 = tile.tan %arg0 : (tensor<8x9xf32>) -> tensor<8x9xf32>
  return %0 : tensor<8x9xf32>
}

// CHECK-LABEL: func.func @sin
func.func @sin(%arg0: tensor<8x9xf32>) -> tensor<8x9xf32> {
  // CHECK: linalg.init_tensor
  // CHECK: linalg.generic
  // CHECK:   sin{{.*}} : f32
  // CHECK:   linalg.yield
  %0 = tile.sin %arg0 : (tensor<8x9xf32>) -> tensor<8x9xf32>
  return %0 : tensor<8x9xf32>
}
