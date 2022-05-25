// RUN: pmlc-opt -convert-linalg-to-pxa %s | FileCheck %s

func @main(%arg0: tensor<3x3x3x3x4xf32> {stdx.const}) -> tensor<5xi32> {
  %cst = arith.constant dense<[3, 3, 3, 3, 4]> : tensor<5xi32>
  return %cst : tensor<5xi32>
}

// CHECK-LABEL: func @main
//  CHECK-SAME: (%[[arg0:.*]]: memref<3x3x3x3x4xf32> {stdx.const}, %[[arg1:.*]]: memref<5xi32>) -> memref<5xi32>
//       CHECK:   %[[t0:.*]] = memref.get_global @cst_memref_0 : memref<5xi32>
//       CHECK:   %[[t1:.*]] = affine.parallel (%[[arg2:.*]]) = (0) to (5) reduce ("assign") -> (memref<5xi32>)
//       CHECK:     %[[t2:.*]] = pxa.load %[[t0]][%[[arg2]]] : memref<5xi32>
//       CHECK:     %[[t3:.*]] = pxa.reduce assign %[[t2]], %[[arg1]][%arg2] : memref<5xi32>
//       CHECK:     affine.yield %[[t3]] : memref<5xi32>
//       CHECK:   return %[[t1]] : memref<5xi32>
