// RUN: pmlc-opt %s | pmlc-opt | FileCheck %s

// CHECK-LABEL: func @dot
func @dot(%A: memref<8x8xf32>, %B: memref<8x8xf32>, %C: memref<8x8xf32>) -> () {
  affine.parallel (%i, %j, %k) = (0, 0, 0) to (8, 8, 8) step (2, 2, 2) {
    // CHECK: xsmm.gemm %{{.*}}[%{{.*}}, %{{.*}}]:8 =
    // CHECK-SAME: %{{.*}}[%{{.*}}, %{{.*}}]:8, %{{.*}}[%{{.*}}, %{{.*}}]:8, [2, 2, 2]
    // CHECK-SAME: memref<8x8xf32>, memref<8x8xf32>, memref<8x8xf32>
    xsmm.gemm %C[%i, %j]:8 = %A[%i, %k]:8, %B[%k, %j]:8, [2, 2, 2]
      : memref<8x8xf32>, memref<8x8xf32>, memref<8x8xf32>
  }
  return
}

func @res2a_branch2a(%I: memref<1x56x56x64xf32>, %K: memref<1x1x64x64xf32>, %O: memref<1x56x56x64xf32>) -> () {
  %c0 = constant 0 : index
  affine.parallel (%x, %y) = (0, 0) to (56, 56) step (14, 1) {
    xsmm.gemm %O[%c0, 14 * %x, %y, %c0]:3584 =
      %K[%c0, %c0, %c0, %c0]:64, %I[%c0, 14 * %x, %y, %c0]:3584, [64, 14, 64]
      : memref<1x56x56x64xf32>, memref<1x1x64x64xf32>, memref<1x56x56x64xf32>
  }
  return
}

func @res2a_branch2b(%I: memref<1x56x56x64xf32>, %K: memref<3x3x64x64xf32>, %O: memref<1x56x56x64xf32>) -> () {
  %c0 = constant 0 : index
  %f0 = constant 0.0 : f32
  %T = alloc() : memref<1x58x58x64xf32>
  linalg.fill(%T, %f0) : memref<1x58x58x64xf32>, f32
  affine.parallel (%x, %y, %k) = (0, 0, 0) to (56, 56, 64) {
    %0 = affine.load %I[%c0, %x, %y, %k] : memref<1x56x56x64xf32>
    affine.store %0, %T[%c0, %x + 1, %y + 1, %k] : memref<1x58x58x64xf32>
  }
  affine.parallel (%x, %y, %kx, %ky) = (0, 0, 0, 0) to (56, 56, 3, 3) step (14, 1, 1, 1) {
    xsmm.gemm %O[%c0, 14 * %x, %y, %c0]:3584 =
      %K[%kx, %ky, %c0, %c0]:64, %T[%c0, 14 * %x + %kx, %y + %ky, %c0]:3712, [64, 14, 64]
      : memref<1x56x56x64xf32>, memref<3x3x64x64xf32>, memref<1x58x58x64xf32>
  }
  return
}
