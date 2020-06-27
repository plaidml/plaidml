// RUN: pmlc-opt %s | pmlc-opt | FileCheck %s

#A_tile = affine_map<(m, k) -> (m, k)>
#B_tile = affine_map<(k, n) -> (k, n)>
#C_tile = affine_map<(m, n) -> (m, n)>

// CHECK-LABEL: func @dot
func @dot(%A: memref<8x8xf32>, %B: memref<8x8xf32>, %C: memref<8x8xf32>) -> () {
  affine.parallel (%i, %j, %k) = (0, 0, 0) to (8, 8, 8) step (2, 2, 2) {
    // CHECK: xsmm.gemm.dispatch [2, 2, 2], [2, 2, 2]
    %1 = xsmm.gemm.dispatch [2, 2, 2], [2, 2, 2]
    // CHECK: xsmm.gemm.invoke %{{.*}}, %{{.*}}[%{{.*}}, %{{.*}}]:#{{.*}} =
    // CHECK-SAME: %{{.*}}[%{{.*}}, %{{.*}}]:#{{.*}}, %{{.*}}[%{{.*}}, %{{.*}}]:#{{.*}}, [2, 2, 2]
    // CHECK-SAME: (memref<8x8xf32>, memref<8x8xf32>) -> memref<8x8xf32>
    %2 = xsmm.gemm.invoke %1, %C[%i, %j]:#C_tile = %A[%i, %k]:#A_tile, %B[%k, %j]:#B_tile, [2, 2, 2]
      : (memref<8x8xf32>, memref<8x8xf32>) -> memref<8x8xf32>
  }
  return
}
