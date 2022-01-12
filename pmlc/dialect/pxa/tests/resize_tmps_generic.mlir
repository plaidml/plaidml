// RUN: pmlc-opt --pxa-resize-tmps %s | FileCheck %s
#map0 = affine_map<(d0, d1, d2) -> (0, d0, 0, d1)>
#map1 = affine_map<(d0, d1, d2, d3, d4) -> (0, d0 * 2 + d3, d4, d2)>
#map2 = affine_map<(d0, d1, d2, d3, d4) -> (d3, d4, d2, d1)>

// CHECK-LABEL: func @resize_tmp_generic
func @resize_tmp_generic(%arg0: memref<64xf32>, %arg1: memref<1x230x230x3xf32>, %arg2: memref<7x7x3x64xf32>, %arg3: memref<1x114x114x64xf32>) -> memref<1x114x114x64xf32> {
  %0 = affine.parallel (%arg4, %arg5) = (0, 0) to (7, 8) reduce ("assign") -> (memref<1x114x114x64xf32>) {
  // CHECK: affine.parallel
    %1 = affine.parallel (%arg6) = (0) to (14) reduce ("assign") -> (memref<1x114x114x64xf32>) {
    // CHECK: affine.parallel
      %2 = memref.alloc() : memref<1x112x112x64xf32>
      // CHECK: memref.alloc() : memref<1x16x1x64xf32>
      %3 = affine.parallel (%arg7, %arg8) = (0, 0) to (16, 64) reduce ("assign") -> (memref<1x112x112x64xf32>) {
      // CHECK: affine.parallel (%[[arg7:.*]], %[[arg8:.*]]) = (0, 0) to (16, 64)
        %6 = pxa.load %arg0[%arg8] : memref<64xf32>
        %7 = pxa.reduce assign %6, %2[0, %arg7 + %arg4 * 16, %arg6 + %arg5 * 14, %arg8] : memref<1x112x112x64xf32>
        // CHECK: pxa.reduce assign {{.*}}, {{.*}}[0, %[[arg7]], 0, %[[arg8]]]
        affine.yield %7 : memref<1x112x112x64xf32>
      }
      %4 = pxa.generic (%3[0, %arg4 * 16, %arg6 + %arg5 * 14, 0]: #map0) <addf> @tpp_gemm(%arg1[0, %arg4 * 32, %arg6 * 2 + %arg5 * 28, 0]: #map1, %arg2[0, 0, 0, 0]: #map2) tile: [16, 64, 3, 1, 7, 7] : (memref<1x230x230x3xf32>, memref<7x7x3x64xf32>) -> memref<1x112x112x64xf32>
      // CHECK: pxa.generic (%3[0, 0, 0, 0]: #map0)
      %5 = affine.parallel (%arg7, %arg8) = (0, 0) to (16, 64) reduce ("assign") -> (memref<1x114x114x64xf32>) {
      // CHECK: affine.parallel (%[[arg7:.*]], %[[arg8:.*]]) = (0, 0) to (16, 64)
        %6 = pxa.load %4[0, %arg7 + %arg4 * 16, %arg6 + %arg5 * 14, %arg8] : memref<1x112x112x64xf32>
        // CHECK: pxa.load {{.*}}[0, %[[arg7]], 0, %[[arg8]]]
        %7 = stdx.relu(%6) : (f32) -> f32
        %8 = pxa.reduce assign %7, %arg3[0, %arg7 + %arg4 * 16 + 1, %arg6 + %arg5 * 14 + 1, %arg8] : memref<1x114x114x64xf32>
        affine.yield %8 : memref<1x114x114x64xf32>
      }
      affine.yield %5 : memref<1x114x114x64xf32>
    }
    affine.yield %1 : memref<1x114x114x64xf32>
  }
  return %0 : memref<1x114x114x64xf32>
}
