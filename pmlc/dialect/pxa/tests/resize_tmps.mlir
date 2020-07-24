// RUN: pmlc-opt -canonicalize -pxa-resize-tmps -canonicalize %s | FileCheck %s

// CHECK-LABEL: func @simple_resize
func @simple_resize(%I: memref<2x3xf32>) -> (memref<2x3xf32>) {
  %O = alloc() : memref<2x3xf32>
  // CHECK: alloc() : memref<2x3xf32>
  %O3 = affine.parallel (%i, %j) = (0, 0) to (2, 3) reduce ("assign") -> (memref<2x3xf32>) {
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
  %O4 = affine.parallel (%i1, %j1) = (0, 0) to (100, 100) step (10, 10) reduce ("assign") -> (memref<100x100xf32>) {
    // CHECK: affine.parallel (%[[i1:.*]], %[[j1:.*]]) = 
    %T = alloc() : memref<100x100xf32>
    // CHECK: alloc() : memref<10x10xf32>
    %O3 = affine.parallel (%i2, %j2) = (0, 0) to (10, 10) reduce ("assign") -> (memref<100x100xf32>) {
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

func @no_resize_return() -> memref<10x10xf32> {
// CHECK-LABEL: func @no_resize_return
  %cst = constant 0.000000e+00 : f32
  %0 = alloc() : memref<10x10xf32>
  // CHECK: alloc() : memref<10x10xf32>
  %1 = affine.parallel (%arg0, %arg1) = (0, 0) to (5, 5) reduce ("assign") -> (memref<10x10xf32>) {
    %2 = pxa.reduce assign %cst, %0[%arg0, %arg1] : memref<10x10xf32>
    affine.yield %2 : memref<10x10xf32>
  }
  return %1 : memref<10x10xf32>
}

#set0 = affine_set<(d0, d1, d2, d3) : (d0 * -30 - d1 + 221 >= 0, d2 * -30 - d3 + 221 >= 0)>
func @no_resize_expand(%arg0: memref<32x30xf32>) {
// CHECK-LABEL: func @no_resize_expand
  %0 = alloc() : memref<1x30x32x8x8x32xf32>
  %1 = alloc() : memref<1x222x222x32xf32>
  // CHECK: alloc() : memref<1x222x222x32xf32>
  %2 = affine.parallel (%arg1, %arg2, %arg3, %arg4, %arg5, %arg6) = (0, 0, 0, 0, 0, 0) to (8, 30, 8, 30, 32, 32) reduce ("assign") -> (memref<1x222x222x32xf32>) {
    %4 = affine.if #set0(%arg1, %arg2, %arg3, %arg4) -> memref<1x222x222x32xf32> {
      %5 = affine.load %0[0, %arg2, %arg6, %arg1, %arg3, %arg5] : memref<1x30x32x8x8x32xf32>
      %6 = affine.load %arg0[%arg6, %arg4] : memref<32x30xf32>
      %7 = mulf %5, %6 : f32
      %8 = pxa.reduce add %7, %1[0, %arg1 * 30 + %arg2, %arg3 * 30 + %arg4, %arg5] : memref<1x222x222x32xf32>
      affine.yield %8 : memref<1x222x222x32xf32>
    } else {
      affine.yield %1 : memref<1x222x222x32xf32>
    }
    affine.yield %4 : memref<1x222x222x32xf32>
  }
  return
}