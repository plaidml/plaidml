// RUN: pmlc-opt -stdx-split-main %s | FileCheck %s

func @main(%arg0: tensor<16x16xf32>, %arg1: tensor<16x16xf32> {stdx.const = 0 : index}) -> tensor<16x16xf32> {
  %0 = addf %arg0, %arg1 : tensor<16x16xf32>
  return %0 : tensor<16x16xf32>
}
// CHECK-LABEL: func @main
//  CHECK-SAME: (%[[arg0:.*]]: tensor<16x16xf32> {stdx.const = 0 : index})
//       CHECK:   stdx.closure(%[[arg1:.*]]: tensor<16x16xf32>) -> tensor<16x16xf32>
//       CHECK:     %[[ret:.*]] = addf %[[arg1]], %[[arg0]] : tensor<16x16xf32>
//       CHECK:     stdx.yield %[[ret]] : tensor<16x16xf32>
//       CHECK:   return

func @after(%arg0: tensor<16x16xf32> {stdx.const = 0 : index}) {
  stdx.closure(%arg1: tensor<16x16xf32>) -> tensor<16x16xf32> {
    %0 = addf %arg0, %arg1 : tensor<16x16xf32>
    stdx.yield %0 : tensor<16x16xf32>
  }
  return
}
// CHECK-LABEL: func @after
//  CHECK-SAME: (%[[arg0:.*]]: tensor<16x16xf32> {stdx.const = 0 : index})
//       CHECK:   stdx.closure(%[[arg1:.*]]: tensor<16x16xf32>) -> tensor<16x16xf32>
//       CHECK:     %[[ret:.*]] = addf %[[arg0]], %[[arg1]] : tensor<16x16xf32>
//       CHECK:     stdx.yield %[[ret]] : tensor<16x16xf32>
//       CHECK:   return

func @null() {
  return
}
// CHECK-LABEL: func @null()
//  CHECK-NEXT: return
