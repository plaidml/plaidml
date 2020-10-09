// RUN: pmlc-opt -convert-tile-to-pxa -canonicalize -cse -split-input-file %s | FileCheck %s

func @const() -> tensor<f32> {
  %cst = "eltwise.sconst"() {value = 3.0 : f64} : () -> tensor<f32>
  return %cst : tensor<f32>
}

// CHECK-LABEL: func @const
// CHECK: constant 3.0{{.*}} : f32
// CHECK: affine.parallel
// CHECK:   pxa.reduce assign
// CHECK:   affine.yield
// CHECK: return %{{.*}} : memref<f32>
