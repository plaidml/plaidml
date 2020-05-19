// RUN: pmlc-opt -canonicalize -autotile-10 %s | FileCheck %s

func @dot(%arg0: memref<100x100xf32>, %arg1: memref<100x100xf32>) -> memref<100x100xf32> {
  %obuf = alloc() : memref<100x100xf32>
  %out = affine.parallel (%arg2, %arg3, %arg4) = (0, 0, 0) to (200, 200, 200) step (5, 5, 5) : memref<100x100xf32> {
    %0 = affine.parallel (%arg5, %arg6, %arg7) = (%arg2, %arg3, %arg4) to (%arg2 + 5, %arg3 + 5, %arg4 + 5) : memref<100x100xf32> {
      %1 = affine.load %arg1[%arg5, %arg7] : memref<100x100xf32>
      %2 = affine.load %arg0[%arg7, %arg6] : memref<100x100xf32>
      %3 = mulf %1, %2 : f32
      %4 = pxa.reduce add %3, %obuf[%arg5, %arg6] : memref<100x100xf32>
      affine.yield %4 : memref<100x100xf32>
    }
    affine.yield %0 : memref<100x100xf32>
  }
  return %out : memref<100x100xf32>
}

// CHECK-LABEL: func @dot
// CHECK-SAME: (%[[arg0:.*]]: memref<100x100xf32>, %[[arg1:.*]]: memref<100x100xf32>) -> memref<100x100xf32>
// CHECK: %[[obuf:.*]] = alloc() : memref<100x100xf32>
// CHECK: %1 = affine.parallel (%[[arg2:.*]], %[[arg3:.*]], %[[arg4:.*]]) = (0, 0, 0) to (200, 200, 200) step (50, 50, 50) {
// CHECK:   %2 = affine.parallel (%[[arg5:.*]], %[[arg6:.*]], %[[arg7:.*]]) = (%[[arg2]], %[[arg3]], %[[arg4]]) to (%[[arg2]] + 10, %[[arg3]] + 10, %[[arg4]] + 10) {
// CHECK:     %3 = affine.parallel (%[[arg8:.*]], %[[arg9:.*]], %[[arg10:.*]]) = (%[[arg5]], %[[arg6]], %[[arg7]]) to (%[[arg5]] + 5, %[[arg6]] + 5, %[[arg7]] + 5) {
// CHECK:       %4 = affine.load %[[arg1]][%[[arg8]], %[[arg10]]] : memref<100x100xf32>
// CHECK:       %5 = affine.load %[[arg0]][%[[arg10]], %[[arg9]]] : memref<100x100xf32>
// CHECK:       %6 = mulf %4, %5 : f32
// CHECK:       %7 = pxa.reduce add %6, %[[obuf]][%[[arg8]], %[[arg9]]] : memref<100x100xf32>
// CHECK:       affine.yield %7 : memref<100x100xf32>
// CHECK:     }
// CHECK:     affine.yield %3 : memref<100x100xf32>
// CHECK:   }
// CHECK:   affine.yield %2 : memref<100x100xf32>
// CHECK: }
// CHECK: return %1 : memref<100x100xf32>
