// RUN: pmlc-opt -convert-tile-to-pxa -canonicalize -cse %s | FileCheck %s

!i32 = type tensor<!eltwise.i32>
module {
  func @bit_right_scalar(%arg0: tensor<3x3x!eltwise.u64>) -> tensor<3x3x!eltwise.u64> {
    %c9 = "eltwise.sconst"() {value = 9 : i64} : () -> !i32
    %0 = "eltwise.bit_shr"(%arg0, %c9) : (tensor<3x3x!eltwise.u64>, !i32) -> tensor<3x3x!eltwise.u64>
    return %0 : tensor<3x3x!eltwise.u64>
  }
}

// CHECK-LABEL: func @bit_right_scalar
// CHECK: pxa.parallel
// CHECK: affine.load
// CHECK: shift_right_unsigned
// CHECK: affine.store
