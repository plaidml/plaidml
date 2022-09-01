// RUN: pmlc-opt %s -x86-tpp-combine \
// RUN:   | FileCheck %s

#map1 = affine_map<(d0, d1) -> ()>
#map4 = affine_map<(d0, d1) -> (d1)>
#map9 = affine_map<(d0, d1) -> (0, 0, 0, d0, d1)>
#map10 = affine_map<(d0, d1, d2) -> (0, 0, 0, d0, d1)>
#map11 = affine_map<(d0, d1, d2, d3) -> (0, d3, 0, d0, d2)>
#map12 = affine_map<(d0, d1, d2, d3) -> (0, d3, 0, 0, d2, d1)>
#map13 = affine_map<(d0, d1) -> (0, 0, 1, d0 + 1, d1)>

memref.global "private" constant @cst_scalar_memref_4 : memref<f32> = dense<0.000000e+00>
// CHECK-LABEL: func @tpp_gemm_relu
//       CHECK:   affine.parallel
//	 CHECK:   affine.parallel
//	 CHECK-NOT:   {{.*}} = memref.alloc
//       CHECK:       pxa.generic (%{{.*}}[0, %{{.*}}, %{{.*}} + %{{.*}} * 4 + 1, 1, 0]: #{{.*}}) <assign> @tpp_gemm_relu(%{{.*}}[0, 0, %{{.*}} + %{{.*}} * 4, 0, 0]: #{{.*}}, %{{.*}}[%{{.*}}, 0, 0, 0, 0, 0]: #{{.*}}, %{{.*}}[%{{.*}} * 32]: #{{.*}}) tile: [56, 32, 32, 1, 2]
func @tpp_gemm_relu(%I: memref<1x2x56x56x32xf32>, %K: memref<2x2x1x1x32x32xf32>, %B: memref<64xf32>, %O: memref<1x2x58x58x32xf32>) -> memref<1x2x58x58x32xf32> {
      %125 = memref.alloc() : memref<1x2x58x58x32xf32>
      %0 = memref.get_global @cst_scalar_memref_4 : memref<f32>
      %126 = affine.parallel (%arg110, %arg111) = (0, 0) to (2, 8) reduce ("assign") -> (memref<1x2x58x58x32xf32>) {
        %251 = affine.parallel (%arg112) = (0) to (58) reduce ("assign") -> (memref<1x2x58x58x32xf32>) {
          %252 = pxa.generic (%125[0, %arg110, %arg112, 0, %arg111 * 4]: #map9) <assign> @tpp_identity(%0[]: #map1) tile: [58, 4] : (memref<f32>) -> memref<1x2x58x58x32xf32>
          affine.yield %252 : memref<1x2x58x58x32xf32>
        }
        affine.yield %251 : memref<1x2x58x58x32xf32>
      }
  %ret = affine.parallel (%arg110, %arg111) = (0, 0) to (2, 14) reduce ("assign") -> (memref<1x2x58x58x32xf32>) {
        %251 = affine.parallel (%arg112) = (0) to (4) reduce ("assign") -> (memref<1x2x58x58x32xf32>) {
          %252 = memref.alloc() : memref<1x1x1x56x32xf32>
          %253 = pxa.generic (%252[0, 0, 0, 0, 0]: #map9) <assign> @tpp_identity(%B[%arg110 * 32]: #map4) tile: [56, 32] : (memref<64xf32>) -> memref<1x1x1x56x32xf32>
          %254 = pxa.generic (%253[0, 0, 0, 0, 0]: #map10) <addf> @tpp_gemm(%I[0, 0, %arg112 + %arg111 * 4, 0, 0]: #map11, %K[%arg110, 0, 0, 0, 0, 0]: #map12) tile: [56, 32, 32, 1, 2] : (memref<1x2x56x56x32xf32>, memref<2x2x1x1x32x32xf32>) -> memref<1x1x1x56x32xf32>
          %255 = pxa.generic (%126[0, %arg110, %arg112 + %arg111 * 4 + 1, 1, 0]: #map13) <assign> @tpp_relu(%254[0, 0, 0, 0, 0]: #map9) tile: [56, 32] : (memref<1x1x1x56x32xf32>) -> memref<1x2x58x58x32xf32>
          memref.dealloc %252 : memref<1x1x1x56x32xf32>
          affine.yield %255 : memref<1x2x58x58x32xf32>
        }
        affine.yield %251 : memref<1x2x58x58x32xf32>
      } 
  return %ret : memref<1x2x58x58x32xf32>
}
