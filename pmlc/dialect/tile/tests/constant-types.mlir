// RUN: pmlc-opt -tile-constant-types -tile-constant-types-floatx=f16 -tile-constant-types-intx=i16 -split-input-file %s | FileCheck %s

!i64 = type tensor<!eltwise.i64>
!f64 = type tensor<!eltwise.f64>
module {
  func @lower_precision(%arg0: tensor<3x3x!eltwise.f32>) -> tensor<3x3x!eltwise.f64> {
    %c1 = "eltwise.sconst"() {value = 1 : i64} : () -> !i64
    %cst = "eltwise.sconst"() {value = 2.000000e+00 : f64} : () -> !f64
    %0 = "eltwise.add"(%arg0, %c1) : (tensor<3x3x!eltwise.f32>, !i64) -> tensor<3x3x!eltwise.f32>
    %1 = "eltwise.add"(%0, %cst) : (tensor<3x3x!eltwise.f32>, !f64) -> tensor<3x3x!eltwise.f64>
    return %1 : tensor<3x3x!eltwise.f64>
  }
}

// CHECK-LABEL: func @lower_precision
// CHECK: eltwise.sconst
// CHECK-SAME: {value = 1 : i64} : () -> !i16
// CHECK: eltwise.sconst
// CHECK-SAME: {value = 2.000000e+00 : f64} : () -> !f16


// -----
