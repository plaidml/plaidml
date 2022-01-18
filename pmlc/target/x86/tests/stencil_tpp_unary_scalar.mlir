// RUN: pmlc-opt -x86-stencil-tpp-unary-scalar %s | FileCheck %s



// CHECK-LABEL: func @stencil_cst_assign
func @stencil_cst_assign(%O: memref<5x5xf32>) -> memref<5x5xf32> {
  %cst_0 = constant 0.000000e+00 : f32
  // CHECK: pxa.generic (%{{.*}}[%{{.*}}, %{{.*}}]: #{{.*}}) <assign> @tpp_identity() tile: [5, 5] : (f32) -> memref<5x5xf32>
  %1 = affine.parallel (%ox, %oy) = (0, 0) to (5, 5) reduce ("assign") -> (memref<5x5xf32>) {
    %4 = pxa.reduce assign %cst_0, %O[%ox,%oy] : memref<5x5xf32>
    affine.yield %4 : memref<5x5xf32>
  }
  return %1 : memref<5x5xf32>
}

