// RUN: pmlc-opt -pxa-autotile-example %s | FileCheck %s

// CHECK-LABEL: @dot0
// CHECK-SAME: (%[[arg0:.*]]: memref<100x100xf32>, %[[arg1:.*]]: memref<100x100xf32>) -> memref<100x100xf32>
func @dot0(%arg0: memref<100x100xf32>, %arg1: memref<100x100xf32>) -> memref<100x100xf32> {
  %obuf = alloc() : memref<100x100xf32>
  // CHECK: %[[obuf:.*]] = alloc() : memref<100x100xf32>
  // CHECK: affine.parallel (%[[arg2:.*]], %[[arg3:.*]], %[[arg4:.*]]) = (0, 0, 0) to (100, 100, 100) step (10, 10, 10)
  // CHECK: affine.parallel (%[[arg5:.*]], %[[arg6:.*]], %[[arg7:.*]]) = (%[[arg2]], %[[arg3]], %[[arg4]]) to (%[[arg2]] + 10, %[[arg3]] + 10, %[[arg4]] + 10)
  %out = affine.parallel (%i, %j, %k) = (0, 0, 0) to (100, 100, 100) reduce ("assign") -> (memref<100x100xf32>) {
    // CHECK: %[[arg1]][%[[arg5]], %[[arg7]]]
    %0 = affine.load %arg1[%i, %k] : memref<100x100xf32>
    // CHECK: %[[arg0]][%[[arg7]], %[[arg6]]]
    %1 = affine.load %arg0[%k, %j] : memref<100x100xf32>
    %2 = mulf %0, %1 : f32
    // CHECK: %[[obuf]][%[[arg5]], %[[arg6]]]
    %3 = pxa.reduce add %2, %obuf[%i, %j] : memref<100x100xf32>
    affine.yield %3 : memref<100x100xf32>
  }
  return %out : memref<100x100xf32>
}

// CHECK-LABEL: @dot1
// CHECK-SAME: (%[[arg0:.*]]: memref<100x100xf32>, %[[arg1:.*]]: memref<100x100xf32>) -> memref<100x100xf32>
func @dot1(%arg0: memref<100x100xf32>, %arg1: memref<100x100xf32>) -> memref<100x100xf32> {
  %obuf = alloc() : memref<100x100xf32>
  // CHECK: %[[obuf:.*]] = alloc() : memref<100x100xf32>
  // CHECK: affine.parallel (%[[arg2:.*]], %[[arg3:.*]], %[[arg4:.*]]) = (0, 0, 0) to (200, 200, 200) step (50, 50, 50)
  %out = affine.parallel (%arg2, %arg3, %arg4) = (0, 0, 0) to (200, 200, 200) step (5, 5, 5) reduce ("assign") -> (memref<100x100xf32>) {
    // CHECK: affine.parallel (%[[arg5:.*]], %[[arg6:.*]], %[[arg7:.*]]) = (%[[arg2]], %[[arg3]], %[[arg4]]) to (%[[arg2]] + 10, %[[arg3]] + 10, %[[arg4]] + 10) 
    // CHECK: affine.parallel (%[[arg8:.*]], %[[arg9:.*]], %[[arg10:.*]]) = (%[[arg5]], %[[arg6]], %[[arg7]]) to (%[[arg5]] + 5, %[[arg6]] + 5, %[[arg7]] + 5)
    %0 = affine.parallel (%arg5, %arg6, %arg7) = (%arg2, %arg3, %arg4) to (%arg2 + 5, %arg3 + 5, %arg4 + 5) reduce ("assign") -> (memref<100x100xf32>) {
      // CHECK: %[[arg1]][%[[arg8]], %[[arg10]]]
      %1 = affine.load %arg1[%arg5, %arg7] : memref<100x100xf32>
      // CHECK: %[[arg0]][%[[arg10]], %[[arg9]]]
      %2 = affine.load %arg0[%arg7, %arg6] : memref<100x100xf32>
      %3 = mulf %1, %2 : f32
      // CHECK: %[[obuf]][%[[arg8]], %[[arg9]]]
      %4 = pxa.reduce add %3, %obuf[%arg5, %arg6] : memref<100x100xf32>
      affine.yield %4 : memref<100x100xf32>
    }
    affine.yield %0 : memref<100x100xf32>
  }
  return %out : memref<100x100xf32>
}
