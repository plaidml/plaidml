// RUN: pmlc-opt -stdx-main-closure %s | FileCheck %s

func @main(%arg0: tensor<16x16xf32>, %arg1: tensor<16x16xf32> {stdx.const}) -> tensor<16x16xf32> {
  %0 = tile.add %arg0, %arg1 : (tensor<16x16xf32>, tensor<16x16xf32>) -> tensor<16x16xf32>
  return %0 : tensor<16x16xf32>
}
// CHECK-LABEL: func @main
//  CHECK-SAME: (%[[arg0:.*]]: tensor<16x16xf32> {stdx.const})
//       CHECK:   stdx.closure(%[[arg1:.*]]: tensor<16x16xf32>) -> tensor<16x16xf32>
//       CHECK:     %[[ret:.*]] = tile.add %[[arg1]], %[[arg0]] : (tensor<16x16xf32>, tensor<16x16xf32>) -> tensor<16x16xf32>
//       CHECK:     stdx.yield %[[ret]] : tensor<16x16xf32>
//       CHECK:   return

func @null() {
  return
}
// CHECK-LABEL: func @null()
//  CHECK-NEXT: return
