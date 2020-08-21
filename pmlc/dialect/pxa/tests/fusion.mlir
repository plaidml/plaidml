// RUN: pmlc-opt -pxa-normalize -canonicalize -pxa-fusion -pxa-normalize -canonicalize %s | FileCheck %s

// CHECK-LABEL: func @simple_fusion
func @simple_fusion(%A: memref<2x3xf32>, %B: memref<2x3xf32>, %C: memref<2x3xf32>, %D: memref<2x3xf32>) {
  %T = alloc() : memref<2x3xf32>
  %4 = affine.parallel (%i, %j) = (0, 0) to (2, 3) reduce ("assign") -> (memref<2x3xf32>) {
    %0 = pxa.load %A[%i, %j] : memref<2x3xf32>
    %1 = pxa.load %B[%i, %j] : memref<2x3xf32>
    %2 = addf %0, %1 : f32
    %3 = pxa.reduce assign %2, %T[%i, %j] : memref<2x3xf32>
    affine.yield %3 : memref<2x3xf32>
  }
  %5 = affine.parallel (%i, %j) = (0, 0) to (2, 3) reduce ("assign") -> (memref<2x3xf32>) {
    %0 = pxa.load %4[%i, %j] : memref<2x3xf32>
    %1 = pxa.load %C[%i, %j] : memref<2x3xf32>
    %2 = mulf %0, %1 : f32
    %3 = pxa.reduce assign %2, %D[%i, %j] : memref<2x3xf32>
    affine.yield %3 : memref<2x3xf32>
  }
  return
  // CHECK: affine.parallel (%{{.*}}, %{{.*}}) = (0, 0) to (2, 3)
  // CHECK: pxa.load
  // CHECK: pxa.load
  // CHECK: addf
  // CHECK: pxa.reduce
  // CHECK-NOT: affine.parallel
  // CHECK: pxa.load
  // CHECK: pxa.load
  // CHECK: mulf
  // CHECK: pxa.reduce
  // CHECK: affine.yield
}
