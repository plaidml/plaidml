// RUN: pmlc-opt %s -split-input-file | pmlc-opt | FileCheck %s

// CHECK-LABEL: func @dot
func @dot(%A: memref<8x8xf32>, %B: memref<8x8xf32>, %C: memref<8x8xf32>) -> () {
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

// -----

// CHECK-LABEL: func @relu
// CHECK-SAME: (%[[IN:.*]]: memref<8x8xf32>, %[[OUT:.*]]: memref<8x8xf32>)
func @relu(%I: memref<8x8xf32>, %O: memref<8x8xf32>) -> () {
  // CHECK: affine.parallel (%[[IX:.*]], %[[JX:.*]]) = (0, 0) to (8, 8) step (2, 2)
  affine.parallel (%i, %j) = (0, 0) to (8, 8) step (2, 2) {
    // CHECK: %[[PTR:.*]] = xsmm.unary.dispatch RELU(f32, [2, 2], 2, 2, 0, 0) : (f32) -> f32
    %1 = xsmm.unary.dispatch RELU(f32, [2, 2], 2, 2, 0, 0) : (f32) -> f32
    // CHECK: xsmm.unary.invoke %[[OUT]][%[[IX]], %[[JX]]] = %[[PTR]](%[[IN]][%[[JX]], %[[IX]]])
    // CHECK-SAME: (memref<8x8xf32>) -> memref<8x8xf32>
    xsmm.unary.invoke %O[%i, %j] = %1(%I[%j, %i]) : (memref<8x8xf32>) -> memref<8x8xf32>
  }
  return
}

// -----

// CHECK-LABEL: func @brgemm
func @brgemm() -> () {
  // CHECK: xsmm.brgemm.offs.dispatch.f32 [16, 64, 3], [6, 64, 64]
  %0 = xsmm.brgemm.offs.dispatch.f32 [16, 64, 3], [6, 64, 64]
  return
}

// -----

// CHECK-LABEL: func @brgemm
func @brgemm(%arg0: memref<7x7x3x64xf32> {stdx.const}) -> () {
  %114 = memref.alloc() : memref<1x230x230x3xf32>
  %171 = memref.alloc() : memref<1x1x16x64xf32>
  %176 = xsmm.brgemm.offs.dispatch.f32 [16, 64, 3], [6, 64, 64]
  %c0 = arith.constant 0 : index
  // CHECK: xsmm.brgemm.offs.invoke.f32 %2, %1[%c0, %c0, %c0, %c0] = %0[%c0, %c0, %c0, %c0], %arg0[%c0, %c0, %c0, %c0], aOffsets = [1, 2], bOffsets = [1, 2], numBatches = 2 : (memref<1x230x230x3xf32>, memref<7x7x3x64xf32>) -> memref<1x1x16x64xf32>
  xsmm.brgemm.offs.invoke.f32 %176, %171[%c0, %c0, %c0, %c0] = %114[%c0, %c0, %c0, %c0], %arg0[%c0, %c0, %c0, %c0], aOffsets = [1, 2], bOffsets = [1, 2], numBatches = 2 : (memref<1x230x230x3xf32>, memref<7x7x3x64xf32>) -> memref<1x1x16x64xf32>
  return 
}

// -----

// CHECK-LABEL: func @binary_dispatch
func @binary_dispatch() -> () {
  // CHECK: xsmm.binary.dispatch ADD(bcast1 0 bcast2 0 ldo 32 ldi2 256 ldi1 256 tile [56, 32] compute f32 func (f32, f32) -> f32) : i64
  %0 = xsmm.binary.dispatch ADD ( bcast1 0 bcast2 0 ldo 32 ldi2 256 ldi1 256 tile [56, 32] compute f32 func (f32, f32) -> f32 ) : i64
  return 
}
