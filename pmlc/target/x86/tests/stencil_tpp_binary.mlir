// RUN: pmlc-opt -x86-stencil-tpp-binary %s | FileCheck %s

// CHECK-LABEL: func @binary_add
func @binary_add(%A: memref<256x256xf32>, %B: memref<256x256xf32>,%C: memref<256x256xf32>) -> memref<256x256xf32> {
  // CHECK: affine.parallel
  %0 = affine.parallel (%ox, %oy) = (0, 0) to (128, 128) reduce ("assign") -> (memref<256x256xf32>) {
    // CHECK: pxa.generic (%{{.*}}[%{{.*}} * 2, %{{.*}} * 2]: #{{.*}}) <assign> @tpp_add(%{{.*}}[%{{.*}} * 2, %{{.*}} * 2, %{{.*}} * 2]: #{{.*}}) tile: [2, 2] : (memref<256x256xf32>, memref<256x256xf32>) -> memref<256x256xf32>
    %1 = affine.parallel (%ix, %iy) = (0, 0) to (2, 2) reduce ("assign") -> (memref<256x256xf32>) {
      %2 = pxa.load %A[%ix + %ox * 2, %iy + %oy * 2] : memref<256x256xf32>
      %3 = pxa.load %B[%ix + %ox * 2, %iy + %oy * 2] : memref<256x256xf32>
      %4 = arith.addf %2, %3 : f32
      %5 = pxa.reduce assign %4, %C[%ix + %ox * 2, %iy + %oy * 2] : memref<256x256xf32>
      affine.yield %5 : memref<256x256xf32>
    }
    affine.yield %1 : memref<256x256xf32>
  }
  return %0 : memref<256x256xf32>
}
