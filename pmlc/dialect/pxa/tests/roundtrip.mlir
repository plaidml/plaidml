// RUN: pmlc-opt -split-input-file %s | pmlc-opt | FileCheck %s

#O_tile = affine_map<(m, n) -> (0, m, 0, n)>
#I_tile = affine_map<(m, k) -> (0, m, 0, k)>
#K_tile = affine_map<(k, n) -> (0, 0, k, n)>

// CHECK-LABEL: func @gemm
func @gemm(%I: memref<1x56x56x64xf32>, %K: memref<1x1x64x64xf32>, %O: memref<1x56x56x64xf32>) -> memref<1x56x56x64xf32> {
  %0 = affine.parallel (%x, %y) = (0, 0) to (56, 56) step (14, 1) reduce ("assign") -> (memref<1x56x56x64xf32>) {
    //      CHECK: pxa.generic (%{{.*}}[0, %{{.*}}, %{{.*}}, 0]: #{{.*}}) <addf>
    // CHECK-SAME: @tpp_gemm(%{{.*}}[0, %{{.*}}, %{{.*}}, 0]: #{{.*}}, %{{.*}}[{{.*}}, %{{.*}}, %{{.*}}, 0]: #{{.*}}) tile: [14, 64, 64]
    // CHECK-SAME: : (memref<1x56x56x64xf32>, memref<1x1x64x64xf32>) -> memref<1x56x56x64xf32>
    %1 = pxa.generic (%O[0, %x, %y, 0]: #O_tile) <addf> @tpp_gemm(%I[0, %x, %y, 0]: #I_tile, %K[0, %x, %y, 0]: #K_tile)
      tile: [14, 64, 64] : (memref<1x56x56x64xf32>, memref<1x1x64x64xf32>) -> memref<1x56x56x64xf32>
    //      CHECK: pxa.generic (%{{.*}}[0, %{{.*}}, %{{.*}}, 0]: #{{.*}}) <addf>
    // CHECK-SAME: @tpp_gemm(%{{.*}}[0, %{{.*}}, %{{.*}}, 0]: #{{.*}}, %{{.*}}[{{.*}}, %{{.*}}, %{{.*}}, 0]: #{{.*}}) tile: [14, 64, 64, 1, 7, 7]
    // CHECK-SAME: : (memref<1x56x56x64xf32>, memref<1x1x64x64xf32>) -> memref<1x56x56x64xf32>
    %2 = pxa.generic (%O[0, %x, %y, 0]: #O_tile) <addf> @tpp_gemm(%I[0, %x, %y, 0]: #I_tile, %K[0, %x, %y, 0]: #K_tile)
      tile: [14, 64, 64, 1, 7, 7] : (memref<1x56x56x64xf32>, memref<1x1x64x64xf32>) -> memref<1x56x56x64xf32>
    affine.yield %1 : memref<1x56x56x64xf32>
  }
  return %0 : memref<1x56x56x64xf32>
}

// -----

#tile = affine_map<(x, y) -> (0, x, 0, y)>

// CHECK-LABEL: func @relu
func @relu(%I: memref<1x56x56x64xf32>, %O: memref<1x56x56x64xf32>) -> memref<1x56x56x64xf32> {
  %0 = affine.parallel (%x, %y) = (0, 0) to (56, 56) step (14, 1) reduce ("assign") -> (memref<1x56x56x64xf32>) {
    //      CHECK: pxa.generic (%{{.*}}[0, %{{.*}}, %{{.*}}, 0]: #{{.*}}) <assign>
    // CHECK-SAME: @tpp_relu(%{{.*}}[0, %{{.*}}, %{{.*}}, 0]: #{{.*}}) tile: [4, 64]
    // CHECK-SAME: : (memref<1x56x56x64xf32>) -> memref<1x56x56x64xf32>
    %1 = pxa.generic (%O[0, %x, %y, 0]: #tile) <assign> @tpp_relu(%I[0, %x, %y, 0]: #tile) tile: [4, 64]
      : (memref<1x56x56x64xf32>) -> memref<1x56x56x64xf32>
    affine.yield %1 : memref<1x56x56x64xf32>
  }
  return %0 : memref<1x56x56x64xf32>
}
