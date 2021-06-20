// RUN: pmlc-opt %s -split-input-file \
// RUN:   -x86-convert-pxa-to-affine \
// RUN:   -loop-invariant-code-motion \
// RUN:   -canonicalize \
// RUN:   | FileCheck %s

#map = affine_map<(i, j) -> (i, j)>

// CHECK-LABEL: func @dot
//       CHECK:   xsmm.gemm.dispatch.f32 [2, 2, 2], [8, 6, 6]
//       CHECK:   affine.for
//       CHECK:     affine.for
//       CHECK:       affine.for
//       CHECK:         xsmm.gemm.invoke.f32
func @dot(%A: memref<4x8xf32>, %B: memref<8x6xf32>, %C: memref<4x6xf32>) -> memref<4x6xf32> {
  %ret = affine.parallel (%i, %j, %k) = (0, 0, 0) to (8, 8, 8) step (2, 2, 2) reduce ("assign") -> (memref<4x6xf32>) {
    %0 = pxa.generic (%C[%i, %j]:#map) <addf> @tpp_gemm(%A[%i, %k]:#map, %B[%k, %j]:#map) tile: [2, 2, 2]
      : (memref<4x8xf32>, memref<8x6xf32>) -> memref<4x6xf32>
    affine.yield %0 : memref<4x6xf32>
  }
  return %ret : memref<4x6xf32>
}

// -----

#O_tile = affine_map<(m, n) -> (0, m, 0, n)>
#I_tile = affine_map<(m, k) -> (0, m, 0, k)>
#K_tile = affine_map<(k, n) -> (0, 0, k, n)>

// CHECK-LABEL: func @res2a_branch2a
//       CHECK:   xsmm.gemm.dispatch.f32 [14, 64, 64], [3584, 64, 3584]
//       CHECK:   affine.for
//       CHECK:     affine.for
//       CHECK:       xsmm.gemm.invoke.f32
func @res2a_branch2a(%I: memref<1x56x56x64xf32>, %K: memref<1x1x64x64xf32>, %O: memref<1x56x56x64xf32>) -> memref<1x56x56x64xf32> {
  %ret = affine.parallel (%x, %y) = (0, 0) to (56, 56) step (14, 1) reduce ("assign") -> (memref<1x56x56x64xf32>) {
    %0 = pxa.generic (%O[0, %x, %y, 0]:#O_tile) <addf> @tpp_gemm(%I[0, %x, %y, 0]:#I_tile, %K[0, %x, %y, 0]:#K_tile) tile: [14, 64, 64]
      : (memref<1x56x56x64xf32>, memref<1x1x64x64xf32>) -> memref<1x56x56x64xf32>
    affine.yield %0 : memref<1x56x56x64xf32>
  }
  return %ret : memref<1x56x56x64xf32>
}

// -----

#tile = affine_map<(x, y) -> (0, x, 0, y)>

// CHECK-LABEL: func @relu
//       CHECK:   xsmm.unary.dispatch RELU(f32, [4, 64], 3584, 3584) : (f32) -> f32
//       CHECK:   affine.for
//       CHECK:     affine.for
//       CHECK:       xsmm.unary.invoke
func @relu(%I: memref<1x56x56x64xf32>, %O: memref<1x56x56x64xf32>) -> memref<1x56x56x64xf32> {
  %0 = affine.parallel (%x, %y) = (0, 0) to (56, 56) step (14, 1) reduce ("assign") -> (memref<1x56x56x64xf32>) {
    %1 = pxa.generic (%O[0, %x, %y, 0]: #tile) <assign> @tpp_relu(%I[0, %x, %y, 0]: #tile) tile: [4, 64]
      : (memref<1x56x56x64xf32>) -> memref<1x56x56x64xf32>
    affine.yield %1 : memref<1x56x56x64xf32>
  }
  return %0 : memref<1x56x56x64xf32>
}

// -----

#map0 = affine_map<(d0, d1) -> (0, d0, 0, d1)>
#map1 = affine_map<(d0, d1, d2, d3) -> (0, d2 + d0, d3, d1)>
#map2 = affine_map<(d0, d1, d2, d3) -> (d2, d3, d0, d1)>

// CHECK-LABEL: func @conv_pad
//       CHECK:   xsmm.gemm.dispatch.f32 [14, 16, 16], [256, 16, 224]
//       CHECK:   affine.for
//       CHECK:     xsmm.gemm.invoke.f32
func @conv_pad(%I: memref<1x16x16x16xf32>, %K: memref<3x3x16x16xf32>, %O: memref<1x14x14x16xf32>) -> memref<1x14x14x16xf32> {
    %1 = affine.parallel (%x) = (0) to (14) reduce ("assign") -> (memref<1x14x14x16xf32>) {
      %2 = pxa.generic (%O[0, 0, %x, 0]:#map0) <addf> @tpp_gemm(%I[0, 0, %x, 0]:#map1, %K[0, 0, 0, 0]:#map2) tile: [14, 16, 16]
        : (memref<1x16x16x16xf32>, memref<3x3x16x16xf32>) -> memref<1x14x14x16xf32>
      affine.yield %2 : memref<1x14x14x16xf32>
    }
    return %1 : memref<1x14x14x16xf32>
}
