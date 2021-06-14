// RUN: pmlc-opt -split-input-file -x86-tpp-patterns %s | FileCheck %s

func @res2a_branch2a(%I: memref<1x56x56x64xf32>, %K: memref<1x1x64x64xf32>, %O: memref<1x56x56x64xf32>) -> memref<1x56x56x64xf32> {
  %cst = constant 0.0 : f32
  %0 = affine.parallel (%arg5, %arg6) = (0, 0) to (56, 2) reduce ("assign") -> (memref<1x56x56x64xf32>) {
    %4 = affine.parallel (%arg7, %arg8, %arg9) = (0, 0, 0) to (56, 32, 64) reduce ("assign") -> (memref<1x56x56x64xf32>) {
      %7 = pxa.load %I[0, %arg7, %arg5, %arg9] : memref<1x56x56x64xf32>
      %8 = pxa.load %K[0, 0, %arg9, %arg8 + %arg6 * 32] : memref<1x1x64x64xf32>
      %9 = mulf %7, %8 : f32
      %10 = pxa.reduce addf %9, %O[0, %arg7, %arg5, %arg8 + %arg6 * 32] : memref<1x56x56x64xf32>
      affine.yield %10 : memref<1x56x56x64xf32>
    } {tags = {stencil = [56, 32, 64]}}
    affine.yield %4 : memref<1x56x56x64xf32>
  }
  return %0 : memref<1x56x56x64xf32>
}
//      CHECK: #[[MAP0:.*]] = affine_map<(d0, d1) -> (0, d0, 0, d1)>
//      CHECK: #[[MAP1:.*]] = affine_map<(d0, d1) -> (0, 0, d0, d1)>
//      CHECK: func @res2a_branch2a
//      CHECK:   affine.parallel (%[[X:.*]], %[[C:.*]]) = (0, 0) to (56, 2) reduce ("assign") -> (memref<1x56x56x64xf32>)
//      CHECK:     pxa.generic (%{{.*}}[0, 0, %[[X]], %[[C]] * 32]: #[[MAP0]]) =
// CHECK-SAME:       @tpp_gemm(%{{.*}}[0, 0, %[[X]], 0]: #[[MAP0]], %{{.*}}[0, 0, 0, %[[C]] * 32]: #[[MAP1]]) tile: [56, 32, 64]
// CHECK-SAME:       : (memref<1x56x56x64xf32>, memref<1x1x64x64xf32>) -> memref<1x56x56x64xf32>

// -----

func @resnet50_conv1(%I: memref<1x230x230x3xf32>, %K: memref<7x7x3x64xf32>, %O: memref<1x16x1x64xf32>) -> memref<1x16x1x64xf32> {
  %5 = affine.parallel (%arg111) = (0) to (56) reduce ("assign") -> (memref<1x16x1x64xf32>) {
    %88 = affine.parallel (%arg112, %arg113) = (0, 0) to (7, 2) reduce ("assign") -> (memref<1x16x1x64xf32>) {
      %93 = affine.parallel (%arg114, %arg115) = (0, 0) to (7, 7) reduce ("assign") -> (memref<1x16x1x64xf32>) {
        %97 = affine.parallel (%arg116, %arg117, %arg118) = (0, 0, 0) to (16, 64, 3) reduce ("assign") -> (memref<1x16x1x64xf32>) {
          %98 = pxa.load %I[0, %arg114 + %arg116 * 2 + %arg112 * 32, %arg115 + %arg113 * 2 + %arg111 * 4, %arg118] : memref<1x230x230x3xf32>
          %99 = pxa.load %K[%arg114, %arg115, %arg118, %arg117] : memref<7x7x3x64xf32>
          %100 = mulf %98, %99 : f32
          %101 = pxa.reduce addf %100, %O[0, %arg116, 0, %arg117] : memref<1x16x1x64xf32>
          affine.yield %101 : memref<1x16x1x64xf32>
        } {tags = {stencil = [16, 64, 3]}}
        affine.yield %97 : memref<1x16x1x64xf32>
      }
      affine.yield %93 : memref<1x16x1x64xf32>
    }
    affine.yield %88 : memref<1x16x1x64xf32>
  }
  return %5 : memref<1x16x1x64xf32>
}

// -----

func @big_dot(%arg1: memref<2048x2048xf32>, %arg2: memref<2048x2048xf32>, %arg3: memref<2048x2048xf32>) -> memref<2048x2048xf32> {
  %cst = constant 0.000000e+00 : f32
  %0 = affine.parallel (%arg4) = (0) to (32) reduce ("assign") -> (memref<2048x2048xf32>) {
    %2 = affine.parallel (%arg5, %arg6) = (0, 0) to (32, 2) reduce ("assign") -> (memref<2048x2048xf32>) {
      %3 = affine.parallel (%arg7) = (0) to (32) reduce ("assign") -> (memref<2048x2048xf32>) {
        %4 = affine.parallel (%arg8, %arg9, %arg10) = (0, 0, 0) to (64, 32, 64) reduce ("assign") -> (memref<2048x2048xf32>) {
          %5 = pxa.load %arg1[%arg8 + %arg5 * 64, %arg10 + %arg7 * 64] : memref<2048x2048xf32>
          %6 = pxa.load %arg2[%arg10 + %arg7 * 64, %arg9 + %arg6 * 32 + %arg4 * 64] : memref<2048x2048xf32>
          %7 = mulf %5, %6 : f32
          %8 = pxa.reduce addf %7, %arg3[%arg8 + %arg5 * 64, %arg9 + %arg6 * 32 + %arg4 * 64] : memref<2048x2048xf32>
          affine.yield %8 : memref<2048x2048xf32>
        } {tags = {stencil = [64, 32, 64]}}
        affine.yield %4 : memref<2048x2048xf32>
      }
      affine.yield %3 : memref<2048x2048xf32>
    }
    affine.yield %2 : memref<2048x2048xf32>
  } {tags = {cpuThread}}
  return %0 : memref<2048x2048xf32>
}
//      CHECK: #[[MAP:.*]] = affine_map<(d0, d1) -> (d0, d1)>
//      CHECK: func @big_dot
//      CHECK:   affine.parallel
//      CHECK:     affine.parallel
//      CHECK:       affine.parallel
//      CHECK:         pxa.generic (%{{.*}}[%{{.*}} * 64, %{{.*}} * 64 + %{{.*}} * 32]: #[[MAP]]) =
// CHECK-SAME:           @tpp_gemm(%{{.*}}[%{{.*}} * 64, %{{.*}} * 64]: #[[MAP]], %{{.*}}[%{{.*}} * 64, %{{.*}} * 64 + %{{.*}} * 32]: #[[MAP]])
// CHECK-SAME:           tile: [64, 32, 64] : (memref<2048x2048xf32>, memref<2048x2048xf32>) -> memref<2048x2048xf32>
