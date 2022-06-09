// RUN: pmlc-opt -x86-stencil-tpp-split %s | FileCheck %s

// CHECK-LABEL: func @main
func @main(%A: memref<256x256xf32>, %B: memref<256x256xf32>,%C: memref<256x256xf32>, %D:memref<256x256xf32>) -> memref<256x256xf32> {
  // CHECK: affine.parallel
  %0 = affine.parallel (%ox, %oy) = (0, 0) to (128, 128) reduce ("assign") -> (memref<256x256xf32>) {
    // CHECK: affine.parallel
    // CHECK: affine.parallel
    // CHECK: affine.parallel
    // CHECK:   pxa.reduce assign
    %1 = affine.parallel (%ix, %iy) = (0, 0) to (2, 2) reduce ("assign") -> (memref<256x256xf32>) {
      %2 = pxa.load %A[%ix + %ox * 2, %iy + %oy * 2] : memref<256x256xf32>
      %3 = pxa.load %B[%ix + %ox * 2, %iy + %oy * 2] : memref<256x256xf32>
      %4 = arith.addf %2, %3 : f32
      %5 = pxa.load %C[%ix+%ox*2, %iy+%oy*2]:memref<256x256xf32>
      %6 = stdx.relu(%5):(f32)->f32
      %7 = arith.addf %4, %6:f32
      %8 = stdx.relu(%7):(f32)->f32
      %9 = pxa.reduce assign %8, %D[%ix + %ox * 2, %iy + %oy * 2] : memref<256x256xf32>
      affine.yield %9 : memref<256x256xf32>
    }
    affine.yield %1 : memref<256x256xf32>

  }
  return %0 : memref<256x256xf32>
}
