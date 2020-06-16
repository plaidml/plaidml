// RUN: pmlc-opt -canonicalize -pxa-resize-tmps -canonicalize %s | FileCheck %s

// CHECK-LABEL: func @simple_resize
func @simple_resize(%I: memref<2x3xf32>) -> (memref<2x3xf32>) {
  %O = alloc() : memref<2x3xf32>
  // CHECK: alloc() : memref<2x3xf32>
  %O3 = affine.parallel (%i, %j) = (0, 0) to (2, 3) : memref<2x3xf32> {
    %T = alloc() : memref<2x3xf32>
    // CHECK: alloc() : memref<1x1xf32>
    %v = affine.load %I[%i, %j] : memref<2x3xf32>
    %sqr = mulf %v, %v : f32
    %T2 = pxa.reduce assign %sqr, %T[%i, %j] : memref<2x3xf32>
    // CHECK: pxa.reduce assign %{{.*}}, %{{.*}}[0, 0] : memref<1x1xf32>
    %sqr2 = affine.load %T2[%i, %j] : memref<2x3xf32>
    // CHECK: affine.load %{{.*}}[0, 0] : memref<1x1xf32>
    %cub = mulf %sqr2, %v : f32
    %O2 = pxa.reduce assign %cub, %O[%i, %j] : memref<2x3xf32>
    affine.yield %O2 : memref<2x3xf32>
  }
  return %O3 : memref<2x3xf32>
}

// CHECK-LABEL: func @inner_indexes
func @inner_indexes(%I: memref<100x100xf32>) -> (memref<100x100xf32>) {
  %O = alloc() : memref<100x100xf32>
  // CHECK: alloc() : memref<100x100xf32>
  %O4 = affine.parallel (%i1, %j1) = (0, 0) to (100, 100) step (10, 10) : memref<100x100xf32> {
    // CHECK: affine.parallel (%[[i1:.*]], %[[j1:.*]]) = 
    %T = alloc() : memref<100x100xf32>
    // CHECK: alloc() : memref<10x10xf32>
    %O3 = affine.parallel (%i2, %j2) = (0, 0) to (10, 10) : memref<100x100xf32> {
      // CHECK: affine.parallel (%[[i2:.*]], %[[j2:.*]]) = 
      %v = affine.load %I[%i1 + %i2, %j1 + %j2] : memref<100x100xf32>
      %sqr = mulf %v, %v : f32
      %T2 = pxa.reduce assign %sqr, %T[%i1 + %i2, %j1 + %j2] : memref<100x100xf32>
      // CHECK: pxa.reduce assign %{{.*}}, %{{.*}}[%[[i2]], %[[j2]]] : memref<10x10xf32>
      %sqr2 = affine.load %T2[%i1 + %i2, %j1 + %j2] : memref<100x100xf32>
      // CHECK: affine.load %{{.*}}[%[[i2]], %[[j2]]] : memref<10x10xf32>
      %cub = mulf %sqr2, %v : f32
      %O2 = pxa.reduce assign %cub, %O[%i1 + %i2, %j1 + %j2] : memref<100x100xf32>
      affine.yield %O2 : memref<100x100xf32>
    }
    affine.yield %O3 : memref<100x100xf32>
  }
  return %O4 : memref<100x100xf32>
}
