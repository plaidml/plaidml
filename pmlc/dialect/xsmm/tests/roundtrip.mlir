// RUN: pmlc-opt %s | pmlc-opt | FileCheck %s

// CHECK-LABEL: func @dot
func @dot(%A: memref<8x8xf32>, %B: memref<8x8xf32>, %C: memref<8x8xf32>) -> () {
  affine.parallel (%i, %j, %k) = (0, 0, 0) to (8, 8, 8) step (2, 2, 2) {
    // CHECK: xsmm.gemm.dispatch [2, 2, 2], [2, 2, 2]
    %1 = xsmm.gemm.dispatch [2, 2, 2], [2, 2, 2]
    // CHECK: xsmm.gemm.invoke %{{.*}}, %{{.*}}[%{{.*}}, %{{.*}}] =
    // CHECK-SAME: %{{.*}}[%{{.*}}, %{{.*}}], %{{.*}}[%{{.*}}, %{{.*}}]
    // CHECK-SAME: (memref<8x8xf32>, memref<8x8xf32>) -> memref<8x8xf32>
    xsmm.gemm.invoke %1, %C[%i, %j] = %A[%i, %k], %B[%k, %j]
      : (memref<8x8xf32>, memref<8x8xf32>) -> memref<8x8xf32>
  }
  return
}
