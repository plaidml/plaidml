// RUN: pmlc-opt %s | pmlc-opt | FileCheck %s

// CHECK-LABEL: func @init
func @init() -> tuple<> {
  // CHECK: stdx.pack() : () -> tuple<>
  %0 = stdx.pack() : () -> tuple<>
  return %0 : tuple<>
}

func @closure(%arg0: tensor<16x16xf32> {stdx.const}) {
  stdx.closure(%arg1: tensor<16x16xf32>) -> tensor<16x16xf32> {
    %0 = addf %arg0, %arg1 : tensor<16x16xf32>
    stdx.yield %0 : tensor<16x16xf32>
  }
  return
}
// CHECK-LABEL: func @closure
//  CHECK-SAME: (%[[arg0:.*]]: tensor<16x16xf32> {stdx.const})
//       CHECK:   stdx.closure(%[[arg1:.*]]: tensor<16x16xf32>) -> tensor<16x16xf32>
//       CHECK:     %[[ret:.*]] = addf %[[arg0]], %[[arg1]] : tensor<16x16xf32>
//       CHECK:     stdx.yield %[[ret]] : tensor<16x16xf32>
//       CHECK:   return
