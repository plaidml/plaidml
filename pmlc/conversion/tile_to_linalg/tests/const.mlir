// RUN: pmlc-opt -convert-tile-to-linalg -cse %s | FileCheck %s

func @const() -> tensor<f32> {
  %cst = tile.constant(3.0 : f64) : tensor<f32>
  return %cst : tensor<f32>
}

// CHECK-LABEL: func @const
// CHECK: constant 3.0{{.*}} : f32
// CHECK: linalg.init_tensor
// CHECK: linalg.generic
// CHECK:   linalg.yield
// CHECK: return %{{.*}} : tensor<f32>
