// RUN: pmlc-opt -convert-tile-to-pxa -cse %s | FileCheck %s

func @const_add(%arg0: tensor<4xsi32> {stdx.const}, %arg1: tensor<4xsi32> {stdx.const}) {
  %0 = tile.add %arg0, %arg1 : (tensor<4xsi32>, tensor<4xsi32>) -> tensor<4xsi32>
  stdx.closure() -> tensor<4xsi32> {
    stdx.yield %0 : tensor<4xsi32>
  }
  return
}

// CHECK-LABEL: func @const_add
//       CHECK:   memref.alloc
//       CHECK:   %[[SUM:.*]] = affine.parallel
//       CHECK:     pxa.load
//       CHECK:     pxa.load
//       CHECK:     addi
//       CHECK:     pxa.reduce
//       CHECK:   stdx.closure(%[[OUT:.*]]: memref<4xi32>) -> memref<4xi32>
//       CHECK:     %[[COPY:.*]] = affine.parallel
//       CHECK:       %[[REG:.*]] = pxa.load %[[SUM]]
//       CHECK:       pxa.reduce assign %[[REG]], %[[OUT]]
//       CHECK:     stdx.yield %[[COPY]]
