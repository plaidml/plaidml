// RUN: pmlc-opt -canonicalize -pxa-convert-mem-op %s | FileCheck %s

// CHECK-LABEL: @load_store
func @load_store(%arg0: memref<5x25x13x13xf32>, %arg1: memref<2xi32>, %arg2: memref<5x25x13x13xf32>) {
// CHECK-SAME: (%[[arg0:.*]]: memref<5x25x13x13xf32>, %[[arg1:.*]]: memref<2xi32>, %[[arg2:.*]]: memref<5x25x13x13xf32>)
  %0 = affine.parallel (%arg3) = (0) to (2) reduce ("assign") -> (memref<5x25x13x13xf32>) {
  // CHECK: affine.parallel (%[[arg3:.*]]) = (0) to (2)
    %1 = index_cast %arg3 : index to i32
    %2 = pxa.reduce assign %1, %arg1[%arg3] : memref<2xi32>
    %3 = affine.parallel (%arg4, %arg5, %arg6) = (0, 0, 0) to (5, 13, 13) reduce ("assign") -> (memref<5x25x13x13xf32>) {
    // CHECK: affine.parallel (%[[arg4:.*]], %[[arg5:.*]], %[[arg6:.*]]) = (0, 0, 0) to (5, 13, 13)
      %4 = index_cast %1 : i32 to index
      %5 = load %arg0[%arg4, %4, %arg5, %arg6] : memref<5x25x13x13xf32>
      // CHECK: %[[r0:.*]] = pxa.load %[[arg0]][%[[arg4]], %[[arg3]], %[[arg5]], %[[arg6]]]
      store %5, %arg2[%arg4, %4, %arg5, %arg6] : memref<5x25x13x13xf32>
      // CHECK: %[[r1:.*]] = pxa.reduce assign %[[r0]], %[[arg2]][%[[arg4]], %[[arg3]], %[[arg5]], %[[arg6]]]
      affine.yield %arg2 : memref<5x25x13x13xf32>
      // CHECK: affine.yield %[[r1]]
    }
    affine.yield %3 : memref<5x25x13x13xf32>
    // CHECK: affine.yield
  }
  return
}
