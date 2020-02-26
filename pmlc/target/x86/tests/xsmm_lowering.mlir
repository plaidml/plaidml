// RUN: pmlc-opt -xsmm %s | FileCheck %s

#tile = affine_map<(i, j, k) -> (2 * i, 2 * j, 2 * k)>

// CHECK-LABEL: func @dot
func @dot(%A: memref<4x8xf32>, %B: memref<8x6xf32>, %C: memref<4x6xf32>) -> () {
  affine.parallel (%i, %j, %k) = (0, 0, 0) to (8, 8, 8) step (2, 2, 2) {
    xsmm.gemm %C[%i, %j]:8 = %A[%i, %k]:8, %B[%k, %j]:8, [2, 2, 2]
      : memref<4x6xf32>, memref<4x8xf32>, memref<8x6xf32>
  }
  return
}
