// RUN: pmlc-opt -canonicalize -autotile-10 %s | FileCheck %s

func @dot(%arg0: memref<100x100xf32>, %arg1: memref<100x100xf32>) -> memref<100x100xf32> {
  %obuf = alloc() : memref<100x100xf32>
  %out = affine.parallel (%i, %j, %k) = (0, 0, 0) to (100, 100, 100) : memref<100x100xf32> {
    %0 = affine.load %arg1[%i, %k] : memref<100x100xf32>
    %1 = affine.load %arg0[%k, %j] : memref<100x100xf32>
    %2 = mulf %0, %1 : f32
    %3 = pxa.reduce add %2, %obuf[%i, %j] : memref<100x100xf32>
    affine.yield %3 : memref<100x100xf32>
  }
  return %out : memref<100x100xf32>
}

// CHECK-LABEL: func @dot
// CHECK-SAME: (%arg0: memref<100x100xf32>, %arg1: memref<100x100xf32>) -> memref<100x100xf32>
// CHECK: %1 = affine.parallel (%[[arg2:.*]], %[[arg3:.*]], %[[arg4:.*]]) = (0, 0, 0) to (100, 100, 100) step (10, 10, 10) {
// CHECK:   %2 = affine.parallel (%[[arg5:.*]], %[[arg6:.*]], %[[arg7:.*]]) = (%[[arg2]], %[[arg3]], %[[arg4]]) to (%[[arg2]] + 10, %[[arg3]] + 10, %[[arg4]] + 10) {
// CHECK:     %3 = affine.load %arg1[%[[arg5]], %[[arg7]]] : memref<100x100xf32>
// CHECK:     %4 = affine.load %arg0[%[[arg7]], %[[arg6]]] : memref<100x100xf32>
// CHECK:     %5 = mulf %3, %4 : f32
// CHECK:     %6 = pxa.reduce add %5, %0[%[[arg5]], %[[arg6]]] : memref<100x100xf32>
// CHECK:     affine.yield %6 : memref<100x100xf32>
// CHECK:   }
// CHECK:   affine.yield %2 : memref<100x100xf32>
// CHECK: }
