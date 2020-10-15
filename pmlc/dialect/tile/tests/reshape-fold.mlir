// RUN: pmlc-opt -canonicalize %s | FileCheck %s

// CHECK-LABEL: @basic
// CHECK-SAME: %[[arg0:.*]]: tensor<4x4xsi32>
// CHECK-NEXT: return %[[arg0]]
func @basic(%arg0: tensor<4x4xsi32>) -> tensor<4x4xsi32> {
  %0 = tile.reshape %arg0 : (tensor<4x4xsi32>) -> tensor<4x4xsi32>
  return %0 : tensor<4x4xsi32>
}

// CHECK-LABEL: @no_folding
func @no_folding(%arg0: tensor<4x4xsi32>) -> tensor<2x8xsi32> {
  %0 = tile.reshape %arg0 : (tensor<4x4xsi32>) -> tensor<2x8xsi32>
  return %0 : tensor<2x8xsi32>
  // CHECK-NEXT: tile.reshape %{{.*}} : (tensor<4x4xsi32>) -> tensor<2x8xsi32>
  // CHECK-NEXT: return %{{.*}}
}
