// RUN: pmlc-opt -pxa-fusion="exactly-match=true" -pxa-normalize -canonicalize %s | FileCheck %s

func @fusion_different_idxs(%A: memref<2x3xf32>, %B: memref<2x3xf32>, %C: memref<2x3x4xf32>, %D: memref<2x3x4xf32>) -> memref<2x3x4xf32> {
  %T = alloc() : memref<2x3xf32>
  %4 = affine.parallel (%i, %j) = (0, 0) to (2, 3) reduce ("assign") -> (memref<2x3xf32>) {
    %0 = pxa.load %A[%i, %j] : memref<2x3xf32>
    %1 = pxa.load %B[%i, %j] : memref<2x3xf32>
    %2 = addf %0, %1 : f32
    %3 = pxa.reduce assign %2, %T[%i, %j] : memref<2x3xf32>
    affine.yield %3 : memref<2x3xf32>
  }
  %5 = affine.parallel (%i, %j, %k) = (0, 0, 0) to (2, 3, 4) reduce ("assign") -> (memref<2x3x4xf32>) {
    %0 = pxa.load %4[%i, %j] : memref<2x3xf32>
    %1 = pxa.load %C[%i, %j, %k] : memref<2x3x4xf32>
    %2 = mulf %0, %1 : f32
    %3 = pxa.reduce assign %2, %D[%i, %j, %k] : memref<2x3x4xf32>
    affine.yield %3 : memref<2x3x4xf32>
  }
  return %5 : memref<2x3x4xf32>
}

// CHECK-LABEL: func @fusion_different_idxs
// CHECK:       affine.parallel (%{{.*}}, %{{.*}}) = (0, 0) to (2, 3)
// CHECK:         pxa.load
// CHECK:         pxa.load
// CHECK:         addf
// CHECK:         pxa.reduce
// CHECK:         affine.yield
// CHECK:       affine.parallel (%{{.*}}, %{{.*}}, %{{.*}}) = (0, 0, 0) to (2, 3, 4)
// CHECK:         pxa.load
// CHECK:         pxa.load
// CHECK:         mulf
// CHECK:         pxa.reduce
// CHECK:         affine.yield
