// RUN: pmlc-opt -convert-tile-to-linalg %s | FileCheck %s

func @pad_eltwise() {
  %cst = tile.constant(0.000000e+00 : f64) : tensor<f32>
  stdx.closure(%arg0: tensor<3xf32>) -> tensor<4xf32> {
    %0 = tile.ident %arg0 {padLower = [0], padType = 1 : i64, padUpper = [1]} : (tensor<3xf32>) -> tensor<3xf32>
    stdx.yield %0 : tensor<3xf32>
  }
  return
}

// CHECK-LABEL: func @pad_eltwise()
// CHECK: stdx.closure
// CHECK:   linalg.init_tensor [3] : tensor<3xf32>
// CHECK:   linalg.generic
// CHECK:     linalg.yield
// CHECK:   -> tensor<3xf32>
// CHECK:   linalg.pad_tensor
// CHECK:     linalg.yield
// CHECK:   : tensor<3xf32> to tensor<4xf32>
// CHECK:   stdx.yield {{.*}} : tensor<4xf32>
