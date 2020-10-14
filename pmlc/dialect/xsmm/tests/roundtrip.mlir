// RUN: pmlc-opt %s | pmlc-opt | FileCheck %s

// CHECK-LABEL: func @dot
func @dot(%A: memref<8x8xf32>, %B: memref<8x8xf32>, %C: memref<8x8xf32>) -> () {
  affine.parallel (%i, %j, %k) = (0, 0, 0) to (8, 8, 8) step (2, 2, 2) {
    // CHECK: xsmm.sgemm.dispatch [2, 2, 2], [2, 2, 2]
    %1 = xsmm.sgemm.dispatch [2, 2, 2], [2, 2, 2]
    // CHECK: xsmm.sgemm.invoke %{{.*}}, %{{.*}}[%{{.*}}, %{{.*}}] =
    // CHECK-SAME: %{{.*}}[%{{.*}}, %{{.*}}], %{{.*}}[%{{.*}}, %{{.*}}]
    // CHECK-SAME: (memref<8x8xf32>, memref<8x8xf32>) -> memref<8x8xf32>
    xsmm.sgemm.invoke %1, %C[%i, %j] = %A[%i, %k], %B[%k, %j]
      : (memref<8x8xf32>, memref<8x8xf32>) -> memref<8x8xf32>
  }
  return
}
