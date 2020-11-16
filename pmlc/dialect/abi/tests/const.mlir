// RUN: pmlc-opt --abi-lower-to-abi %s | FileCheck %s

module @const_add {
  func @main(%arg0: memref<4xi32> {tile.const = 0 : index}, %arg1: memref<4xi32> {tile.const = 1 : index}, %arg2: memref<4xi32>) {
    %c4 = constant 4 : index
    omp.parallel num_threads(%c4 : index) default(shared) {
      %0 = call @plaidml_rt_thread_num() : () -> index
      %1 = load %arg0[%0] : memref<4xi32>
      %2 = load %arg1[%0] : memref<4xi32>
      %3 = addi %1, %2 : i32
      store %3, %arg2[%0] : memref<4xi32>
      omp.terminator
    }
    return
  }
  func @plaidml_rt_thread_num() -> index
}

// CHECK-LABEL: module @const_add {
//  CHECK-NEXT:   abi.loop init  {
//  CHECK-NEXT:   ^bb0(%arg0: !llvm.ptr<i8>, %arg1: memref<4xi32>, %arg2: memref<4xi32>):  // no predecessors
//  CHECK-NEXT:     abi.yield %arg1, %arg2 : memref<4xi32>, memref<4xi32>
//  CHECK-NEXT:   } body  {
//  CHECK-NEXT:   ^bb0(%arg0: memref<4xi32>, %arg1: memref<4xi32>, %arg2: memref<4xi32>):  // no predecessors
//  CHECK-NEXT:     %[[C4:.*]] = constant 4 : index
//  CHECK-NEXT:     omp.parallel num_threads(%[[C4]] : index) default(shared) {
//  CHECK-NEXT:       %[[TID:.*]] = call @plaidml_rt_thread_num() : () -> index
//  CHECK-NEXT:       %[[V1:.*]] = load %arg0[%[[TID]]] : memref<4xi32>
//  CHECK-NEXT:       %[[V2:.*]] = load %arg1[%[[TID]]] : memref<4xi32>
//  CHECK-NEXT:       %[[SUM:.*]] = addi %[[V1]], %[[V2]] : i32
//  CHECK-NEXT:       store %[[SUM]], %arg2[%[[TID]]] : memref<4xi32>
//  CHECK-NEXT:       omp.terminator
//  CHECK-NEXT:     }
//  CHECK-NEXT:     abi.terminator
//  CHECK-NEXT:   } fini  {
//  CHECK-NEXT:   ^bb0(%arg0: memref<4xi32>, %arg1: memref<4xi32>):  // no predecessors
//  CHECK-NEXT:     abi.terminator
//  CHECK-NEXT:   }
//  CHECK-NEXT:   func @plaidml_rt_thread_num() -> index
//  CHECK-NEXT: }
