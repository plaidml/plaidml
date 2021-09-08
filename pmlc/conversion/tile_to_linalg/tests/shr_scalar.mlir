// RUN: pmlc-opt -convert-tile-to-linalg -cse %s | FileCheck %s

module {
  func @bit_right_scalar(%arg0: tensor<3x3xui64>) -> tensor<3x3xui64> {
    %c9 = tile.constant(9 : i64) : tensor<si32>
    %0 = tile.bit_shr %arg0, %c9 : (tensor<3x3xui64>, tensor<si32>) -> tensor<3x3xui64>
    return %0 : tensor<3x3xui64>
  }
}

// CHECK-LABEL: func @bit_right_scalar
// CHECK: linalg.init_tensor [3, 3]
// CHECK: linalg.generic
// CHECK:   sexti
// CHECK:   shift_right_unsigned
// CHECK:   linalg.yield
// CHECK: return
