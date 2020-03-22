// RUN: pmlc-opt -tile-constant-types -tile-constant-types-floatx=f64 -tile-constant-types-intx=u64 -canonicalize -split-input-file %s | FileCheck %s

!i64 = type tensor<!eltwise.i64>
module {
  func @higher_precision_constants(%arg0: tensor<3x3xf32>) -> tensor<3x3xf32> {
    %c1 = "eltwise.sconst"() {value = 1 : i64} : () -> !i64
    %cst = "eltwise.sconst"() {value = 2.000000e+00 : f64} : () -> !f32
    %0 = "eltwise.add"(%arg0, %c1) : (tensor<3x3xf32>, !i64) -> tensor<3x3xf32>
    %1 = "eltwise.add"(%0, %cst) : (tensor<3x3xf32>, !f32) -> tensor<3x3xf32>
    return %1 : tensor<3x3xf32>
  }
}

// CHECK-LABEL: func @higher_precision_constants
// CHECK: eltwise.sconst
// CHECK-SAME: {value = 1 : i64} : () -> !u64
// CHECK: eltwise.sconst
// CHECK-SAME: {value = 2.000000e+00 : f64} : () -> !f64
// CHECK: return
// CHECK-SAME: tensor<3x3x!eltwise.f64>
