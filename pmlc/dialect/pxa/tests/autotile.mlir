// RUN: pmlc-opt -canonicalize -autotile-10 %s | FileCheck %s

// CHECK-LABEL: @dot
func @dot(%arg0: memref<100x100xf32>, %arg1: memref<100x100xf32>, %arg2: memref<100x100xf32>) {
  affine.parallel (%i, %j, %k) = (0, 0, 0) to (100, 100, 100) {
    %0 = affine.load %arg1[%i, %k] : memref<100x100xf32>
    %1 = affine.load %arg0[%k, %j] : memref<100x100xf32>
    %2 = mulf %0, %1 : f32
    pxa.reduce add %2, %arg2[%i, %j] : memref<100x100xf32>
  }
  return
}
// CHECK: affine.parallel
// CHECK: affine.parallel
