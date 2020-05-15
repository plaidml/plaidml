// RUN: pmlc-opt -canonicalize -autotile-10 %s | FileCheck %s

// CHECK-LABEL: @dot
func @dot(%arg0: memref<100x100xf32>, %arg1: memref<100x100xf32>) -> memref<100x100xf32> {
  %obuf = alloc() : memref<100x100xf32>
  %out = affine.parallel (%i, %j, %k) = (0, 0, 0) to (100, 100, 100) : memref<100x100xf32> {
    %0 = affine.load %arg1[%i, %k] : memref<100x100xf32>
    %1 = affine.load %arg0[%k, %j] : memref<100x100xf32>
    %2 = mulf %0, %1 : f32
    %3 = pxa.reduce add %2, %obuf[%i, %j] : memref<100x100xf32>
    affine.yield %3 : memref<100x100xf32>
  }
  return %out : memref<100x100xf32>
}
// CHECK: affine.parallel
// CHECK: affine.parallel
