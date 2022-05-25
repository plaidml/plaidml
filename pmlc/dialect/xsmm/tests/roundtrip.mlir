// RUN: pmlc-opt %s | pmlc-opt | FileCheck %s

// CHECK-LABEL: func.func @dot
func.func @dot(%A: memref<8x8xf32>, %B: memref<8x8xf32>, %C: memref<8x8xf32>) -> () {
  affine.parallel (%i, %j, %k) = (0, 0, 0) to (8, 8, 8) step (2, 2, 2) {
    // CHECK: xsmm.gemm.dispatch.f32 [2, 2, 2], [2, 2, 2]
    %1 = xsmm.gemm.dispatch.f32 [2, 2, 2], [2, 2, 2]
    // CHECK: xsmm.gemm.invoke.f32 %{{.*}}, %{{.*}}[%{{.*}}, %{{.*}}] =
    // CHECK-SAME: %{{.*}}[%{{.*}}, %{{.*}}], %{{.*}}[%{{.*}}, %{{.*}}]
    // CHECK-SAME: (memref<8x8xf32>, memref<8x8xf32>) -> memref<8x8xf32>
    xsmm.gemm.invoke.f32 %1, %C[%i, %j] = %A[%i, %k], %B[%k, %j]
      : (memref<8x8xf32>, memref<8x8xf32>) -> memref<8x8xf32>
  }
  return
}


// CHECK-LABEL: func.func @relu
// CHECK-SAME: (%[[IN:.*]]: memref<8x8xf32>, %[[OUT:.*]]: memref<8x8xf32>)
func.func @relu(%I: memref<8x8xf32>, %O: memref<8x8xf32>) -> () {
  // CHECK: affine.parallel (%[[IX:.*]], %[[JX:.*]]) = (0, 0) to (8, 8) step (2, 2)
  affine.parallel (%i, %j) = (0, 0) to (8, 8) step (2, 2) {
    // CHECK: %[[PTR:.*]] = xsmm.unary.dispatch RELU(f32, [2, 2], 2, 2, 0) : (f32) -> f32
    %1 = xsmm.unary.dispatch RELU(f32, [2, 2], 2, 2, 0) : (f32) -> f32
    // CHECK: xsmm.unary.invoke %[[OUT]][%[[IX]], %[[JX]]] = %[[PTR]](%[[IN]][%[[JX]], %[[IX]]])
    // CHECK-SAME: (memref<8x8xf32>) -> memref<8x8xf32>
    xsmm.unary.invoke %O[%i, %j] = %1(%I[%j, %i]) : (memref<8x8xf32>) -> memref<8x8xf32>
  }
  return
}
