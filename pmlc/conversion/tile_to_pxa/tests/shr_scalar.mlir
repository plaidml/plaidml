// RUN: pmlc-opt -convert-tile-to-pxa -canonicalize -cse %s | FileCheck %s

module {
  func @bit_right_scalar(%arg0: tensor<3x3xui64>) -> tensor<3x3xui64> {
    %c9 = "eltwise.sconst"() {value = 9 : i64} : () -> si32
    %0 = "eltwise.bit_shr"(%arg0, %c9) : (tensor<3x3xui64>, si32) -> tensor<3x3xui64>
    return %0 : tensor<3x3xui64>
  }
}

// CHECK-LABEL: func @bit_right_scalar
// CHECK: affine.parallel
// CHECK: pxa.load
// CHECK: shift_right_unsigned
// CHECK: pxa.reduce assign
