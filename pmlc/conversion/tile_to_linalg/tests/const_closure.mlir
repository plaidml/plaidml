// RUN: pmlc-opt -convert-tile-to-linalg %s | FileCheck %s

func @main(%arg0: tensor<4xsi32> {stdx.const}, %arg1: tensor<4xsi32> {stdx.const}) {
  %0 = tile.add %arg0, %arg1 : (tensor<4xsi32>, tensor<4xsi32>) -> tensor<4xsi32>
  stdx.closure() -> tensor<4xsi32> {
    stdx.yield %0 : tensor<4xsi32>
  }
  return
}

// CHECK-LABEL: func @main
// CHECK: linalg.generic
// CHECK:   addi
// CHECK:   linalg.yield
// CHECK: stdx.closure() -> tensor<4xi32>
// CHECK:   linalg.generic
// CHECK:     linalg.yield
// CHECK:   stdx.yield
// CHECK: return
