// RUN: pmlc-opt -canonicalize -pxa-resize-tmps -canonicalize %s | FileCheck %s

// CHECK-LABEL: func @simple_fusion
func @simple_fusion(%I: memref<2x3xf32>) -> (memref<2x3xf32>) {
  %O = alloc() : memref<2x3xf32>
  %O3 = affine.parallel (%i, %j) = (0, 0) to (2, 3) : memref<2x3xf32> {
    %T = alloc() : memref<2x3xf32>
    %v = affine.load %I[%i, %j] : memref<2x3xf32>
    %sqr = mulf %v, %v : f32
    %T2 = pxa.reduce assign %sqr, %T[%i, %j] : memref<2x3xf32>
    %sqr2 = affine.load %T2[%i, %j] : memref<2x3xf32>
    %cub = mulf %sqr2, %v : f32
    %O2 = pxa.reduce assign %cub, %O[%i, %j] : memref<2x3xf32>
    affine.yield %O2 : memref<2x3xf32>
  }
  return %O3 : memref<2x3xf32>
}

