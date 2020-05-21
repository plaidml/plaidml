// RUN: pmlc-opt -canonicalize %s | FileCheck %s

// CHECK-LABEL: @basic
// CHECK-SAME: %[[arg0:.*]]: tensor<4x4xsi32>
// CHECK-NEXT: return %[[arg0]]
func @basic(%arg0: tensor<4x4xsi32>) -> tensor<4x4xsi32> {
  %c4 = tile.constant 4
  %0 = "tile.reshape"(%arg0, %c4, %c4) : (tensor<4x4xsi32>, index, index) -> tensor<4x4xsi32>
  return %0 : tensor<4x4xsi32>
}

// CHECK-LABEL: @no_folding
func @no_folding(%arg0: tensor<4x4xsi32>) -> tensor<2x8xsi32> {
  %c2 = tile.constant 2
  %c8 = tile.constant 8
  %0 = "tile.reshape"(%arg0, %c2, %c8) : (tensor<4x4xsi32>, index, index) -> tensor<2x8xsi32>
  return %0 : tensor<2x8xsi32>
  // CHECK-NEXT: %c2 = tile.constant 2
  // CHECK-NEXT: %c8 = tile.constant 8
  // CHECK-NEXT: %0 = "tile.reshape"(%arg0, %c2, %c8) : (tensor<4x4xsi32>, index, index) -> tensor<2x8xsi32>
  // CHECK-NEXT: return %0
}
