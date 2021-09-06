// RUN: pmlc-opt -convert-linalg-to-pxa %s | FileCheck %s

func @test_copy(%arg0: memref<16x16xf32>, %arg1: memref<16x16xf32>) {
  linalg.copy(%arg0, %arg1) : memref<16x16xf32>, memref<16x16xf32>
  return
}

// CHECK-LABEL: func @test_copy
// CHECK-SAME: (%[[arg0:.*]]: memref<16x16xf32>, %[[arg1:.*]]: memref<16x16xf32>)
// CHECK: affine.parallel (%[[arg2:.*]], %[[arg3:.*]]) = (0, 0) to (16, 16) reduce ("assign") -> (memref<16x16xf32>)
// CHECK:   %[[t0:.*]] = pxa.load %[[arg0]][%[[arg2]], %[[arg3]]] : memref<16x16xf32>
// CHECK:   %[[t1:.*]] = pxa.reduce assign %[[t0]], %[[arg1]][%[[arg2]], %[[arg3]]] : memref<16x16xf32>
// CHECK:   affine.yield %[[t1]] : memref<16x16xf32>
// CHECK: return
