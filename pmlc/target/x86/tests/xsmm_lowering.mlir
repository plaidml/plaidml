// RUN: pmlc-opt -xsmm -split-input-file %s | FileCheck %s

#A_tile = affine_map<(m, n, k) -> (m, k)>
#B_tile = affine_map<(m, n, k) -> (k, n)>
#C_tile = affine_map<(m, n, k) -> (m, n)>

// CHECK-LABEL: func @dot
func @dot(%A: memref<4x8xf32>, %B: memref<8x6xf32>, %C: memref<4x6xf32>) -> () {
  affine.parallel (%i, %j, %k) = (0, 0, 0) to (8, 8, 8) step (2, 2, 2) {
    xsmm.gemm %C[%i, %j]:#C_tile = %A[%i, %k]:#A_tile, %B[%k, %j]:#B_tile, [2, 2, 2]
      : memref<4x6xf32>, memref<4x8xf32>, memref<8x6xf32>
  }
  return
}

// -----

#O_tile = affine_map<(m, n, k) -> (0, n, 0, m)>
#K_tile = affine_map<(m, n, k) -> (0, 0, m, k)>
#I_tile = affine_map<(m, n, k) -> (0, n, 0, k)>

func @res2a_branch2a(%I: memref<1x56x56x64xf32>, %K: memref<1x1x64x64xf32>, %O: memref<1x56x56x64xf32>) -> () {
  %c0 = constant 0 : index
  affine.parallel (%x, %y) = (0, 0) to (56, 56) step (14, 1) {
    xsmm.gemm %O[%c0, 14 * %x, %y, %c0]:#O_tile
      = %K[%c0, %c0, %c0, %c0]:#K_tile, %I[%c0, 14 * %x, %y, %c0]:#I_tile, [64, 14, 64]
      : memref<1x56x56x64xf32>, memref<1x1x64x64xf32>, memref<1x56x56x64xf32>
  }
  return
}
