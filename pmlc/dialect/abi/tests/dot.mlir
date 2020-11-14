// RUN: pmlc-opt --abi-lower-to-abi %s | FileCheck %s

module @dot {
  func @main(%arg0: memref<8x16xf32>, %arg1: memref<16x32xf32>, %arg2: memref<8x32xf32>) {
    %cst = constant 0.000000e+00 : f32
    %c0 = constant 0 : index
    %c1 = constant 1 : index
    %c2 = constant 2 : index
    %c3 = constant 3 : index
    %c5 = constant 5 : index
    %c6 = constant 6 : index
    %c8 = constant 8 : index
    %c7 = constant 7 : index
    %c4 = constant 4 : index
    omp.parallel num_threads(%c4 : index) default(shared) {
      %1 = call @plaidml_rt_thread_num() : () -> index
      br ^bb1(%c0 : index)
    ^bb1(%2: index):  // 2 preds: ^bb0, ^bb2
      %3 = cmpi "slt", %2, %c8 : index
      cond_br %3, ^bb2, ^bb3
    ^bb2:  // pred: ^bb1
      %4 = muli %1, %c8 : index
      store %cst, %arg2[%2, %4] : memref<8x32xf32>
      %5 = addi %4, %c1 : index
      store %cst, %arg2[%2, %5] : memref<8x32xf32>
      %6 = addi %4, %c2 : index
      store %cst, %arg2[%2, %6] : memref<8x32xf32>
      %7 = addi %4, %c3 : index
      store %cst, %arg2[%2, %7] : memref<8x32xf32>
      %8 = addi %4, %c4 : index
      store %cst, %arg2[%2, %8] : memref<8x32xf32>
      %9 = addi %4, %c5 : index
      store %cst, %arg2[%2, %9] : memref<8x32xf32>
      %10 = addi %4, %c6 : index
      store %cst, %arg2[%2, %10] : memref<8x32xf32>
      %11 = addi %4, %c7 : index
      store %cst, %arg2[%2, %11] : memref<8x32xf32>
      %12 = addi %2, %c1 : index
      br ^bb1(%12 : index)
    ^bb3:  // pred: ^bb1
      omp.terminator
    }
    %0 = xsmm.gemm.dispatch.f32 [8, 32, 16], [16, 32, 32]
    xsmm.gemm.invoke.f32 %0, %arg2[%c0, %c0] = %arg0[%c0, %c0], %arg1[%c0, %c0] : (memref<8x16xf32>, memref<16x32xf32>) -> memref<8x32xf32>
    return
  }
  func @plaidml_rt_thread_num() -> index
}

// CHECK-LABEL: module @dot {
//  CHECK-NEXT:   abi.loop init  {
//  CHECK-NEXT:   ^bb0(%arg0: !llvm.ptr<i8>):  // no predecessors
//  CHECK-NEXT:     abi.yield
//  CHECK-NEXT:   } yield [] body  {
//  CHECK-NEXT:   ^bb0(%arg0: memref<8x16xf32>, %arg1: memref<16x32xf32>, %arg2: memref<8x32xf32>):  // no predecessors
//  CHECK-NEXT:     %[[CFZERO:.*]] = constant 0.000000e+00 : f32
//  CHECK-NEXT:     %[[C0:.*]] = constant 0 : index
//  CHECK-NEXT:     %[[C1:.*]] = constant 1 : index
//  CHECK-NEXT:     %[[C2:.*]] = constant 2 : index
//  CHECK-NEXT:     %[[C3:.*]] = constant 3 : index
//  CHECK-NEXT:     %[[C5:.*]] = constant 5 : index
//  CHECK-NEXT:     %[[C6:.*]] = constant 6 : index
//  CHECK-NEXT:     %[[C8:.*]] = constant 8 : index
//  CHECK-NEXT:     %[[C7:.*]] = constant 7 : index
//  CHECK-NEXT:     %[[C4:.*]] = constant 4 : index
//  CHECK-NEXT:     omp.parallel num_threads(%[[C4]] : index) default(shared) {
//  CHECK-NEXT:       %1 = call @plaidml_rt_thread_num() : () -> index
//  CHECK-NEXT:       br ^bb1(%[[C0]] : index)
//  CHECK-NEXT:     ^bb1(%[[MAJOR_IDX:.*]]: index):  // 2 preds: ^bb0, ^bb2
//  CHECK-NEXT:       %[[DONE:.*]] = cmpi "slt", %[[MAJOR_IDX]], %[[C8]] : index
//  CHECK-NEXT:       cond_br %[[DONE]], ^bb2, ^bb3
//  CHECK-NEXT:     ^bb2:  // pred: ^bb1
//  CHECK-NEXT:       %[[MINOR_IDX_BASE:.*]] = muli %1, %[[C8]] : index
//  CHECK-NEXT:       store %[[CFZERO]], %arg2[%[[MAJOR_IDX]], %[[MINOR_IDX_BASE]]] : memref<8x32xf32>
//  CHECK-NEXT:       %[[MINOR_IDX_1:.*]] = addi %[[MINOR_IDX_BASE]], %[[C1]] : index
//  CHECK-NEXT:       store %[[CFZERO]], %arg2[%[[MAJOR_IDX]], %[[MINOR_IDX_1]]] : memref<8x32xf32>
//  CHECK-NEXT:       %[[MINOR_IDX_2:.*]] = addi %[[MINOR_IDX_BASE]], %[[C2]] : index
//  CHECK-NEXT:       store %[[CFZERO]], %arg2[%[[MAJOR_IDX]], %[[MINOR_IDX_2]]] : memref<8x32xf32>
//  CHECK-NEXT:       %[[MINOR_IDX_3:.*]] = addi %[[MINOR_IDX_BASE]], %[[C3]] : index
//  CHECK-NEXT:       store %[[CFZERO]], %arg2[%[[MAJOR_IDX]], %[[MINOR_IDX_3]]] : memref<8x32xf32>
//  CHECK-NEXT:       %[[MINOR_IDX_4:.*]] = addi %[[MINOR_IDX_BASE]], %[[C4]] : index
//  CHECK-NEXT:       store %[[CFZERO]], %arg2[%[[MAJOR_IDX]], %[[MINOR_IDX_4]]] : memref<8x32xf32>
//  CHECK-NEXT:       %[[MINOR_IDX_5:.*]] = addi %[[MINOR_IDX_BASE]], %[[C5]] : index
//  CHECK-NEXT:       store %[[CFZERO]], %arg2[%[[MAJOR_IDX]], %[[MINOR_IDX_5]]] : memref<8x32xf32>
//  CHECK-NEXT:       %[[MINOR_IDX_6:.*]] = addi %[[MINOR_IDX_BASE]], %[[C6:.*]] : index
//  CHECK-NEXT:       store %[[CFZERO]], %arg2[%[[MAJOR_IDX]], %[[MINOR_IDX_6]]] : memref<8x32xf32>
//  CHECK-NEXT:       %[[MINOR_IDX_7:.*]] = addi %[[MINOR_IDX_BASE]], %[[C7]] : index
//  CHECK-NEXT:       store %[[CFZERO]], %arg2[%[[MAJOR_IDX]], %[[MINOR_IDX_7]]] : memref<8x32xf32>
//  CHECK-NEXT:       %[[NEW_MAJOR_IDX:.*]] = addi %[[MAJOR_IDX]], %[[C1]] : index
//  CHECK-NEXT:       br ^bb1(%[[NEW_MAJOR_IDX]] : index)
//  CHECK-NEXT:     ^bb3:  // pred: ^bb1
//  CHECK-NEXT:       omp.terminator
//  CHECK-NEXT:     }
//  CHECK-NEXT:     %[[FUNC:.*]] = xsmm.gemm.dispatch.f32 [8, 32, 16], [16, 32, 32]
//  CHECK-NEXT:     xsmm.gemm.invoke.f32 %[[FUNC]], %arg2[%[[C0]], %[[C0]]] = %arg0[%[[C0]], %[[C0]]], %arg1[%[[C0]], %[[C0]]] : (memref<8x16xf32>, memref<16x32xf32>) -> memref<8x32xf32>
//  CHECK-NEXT:     abi.terminator
//  CHECK-NEXT:   } fini  {
//  CHECK-NEXT:     abi.terminator
//  CHECK-NEXT:   }
//  CHECK-NEXT:   func @plaidml_rt_thread_num() -> index
//  CHECK-NEXT: }
