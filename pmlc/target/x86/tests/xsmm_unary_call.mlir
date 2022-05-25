// RUN: pmlc-opt %s \
// RUN:     -x86-convert-pxa-to-affine \
// RUN:     -lower-affine \
// RUN:     -convert-scf-to-openmp \
// RUN:     -convert-arith-to-llvm \ 
// RUN:     -convert-memref-to-llvm \ 
// RUN:     -convert-openmp-to-llvm \
// RUN:     -canonicalize \
// RUN:     -x86-convert-std-to-llvm \ 
// RUN:     -canonicalize -reconcile-unrealized-casts \
// RUN:   | pmlc-jit | FileCheck %s

!eltwise = type memref<8x3xf32>

func.func private @printMemrefF32(memref<*xf32>) attributes { llvm.emit_c_interface }

func.func @fill_2d(%arg0: memref<?x?xf32>, %arg1: i1) {
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %c5 = arith.constant 5 : index
    %0 = memref.dim %arg0, %c0 : memref<?x?xf32>
    %1 = memref.dim %arg0, %c1 : memref<?x?xf32>
    %c0_0 = arith.constant 0 : index
    %c0_1 = arith.constant 0 : index
    %c1_2 = arith.constant 1 : index
    %c1_3 = arith.constant 1 : index
    scf.parallel (%arg2, %arg3) = (%c0_0, %c0_1) to (%0, %1) step (%c1_2, %c1_3) {
      %2 = arith.muli %arg2, %1 : index
      %3 = arith.addi %2, %arg3 : index
      %4 = arith.select %arg1, %3, %c0 : index
      %5 = arith.addi %arg2, %arg3 : index
      %6 = arith.addi %5, %4 : index
      %7 = arith.subi %6, %c5 : index
      %8 = arith.index_cast %7 : index to i64
      %9 = arith.sitofp %8 : i64 to f32
      memref.store %9, %arg0[%arg2, %arg3] : memref<?x?xf32>
      scf.yield
    }
    return
}

func.func @main() attributes { llvm.emit_c_interface } {
  %false = arith.constant 0 : i1
  %true = arith.constant 1 : i1
  %A = memref.alloc() : !eltwise
  %A_2d = memref.cast %A : !eltwise to memref<?x?xf32>
  %A_ud = memref.cast %A : !eltwise to memref<*xf32>
  call @fill_2d(%A_2d, %false) : (memref<?x?xf32>, i1) -> ()
  %B = memref.alloc() : !eltwise
  %B_2d = memref.cast %B : !eltwise to memref<?x?xf32>
  %B_ud = memref.cast %B : !eltwise to memref<*xf32>
  call @fill_2d(%B_2d, %true) : (memref<?x?xf32>, i1) -> ()
  call @printMemrefF32(%A_ud) : (memref<*xf32>) -> ()
  // CHECK:  [-5,   -4,   -3],
  // CHECK:  [-4,   -3,   -2],
  // CHECK:  [-3,   -2,   -1],
  // CHECK:  [-2,   -1,   0],
  // CHECK:  [-1,   0,   1],
  // CHECK:  [0,   1,   2],
  // CHECK:  [1,   2,   3],
  // CHECK:  [2,   3,   4]
  call @printMemrefF32(%B_ud) : (memref<*xf32>) -> ()
  // CHECK:  [-5,   -3,   -1],
  // CHECK:  [-1,   1,   3],
  // CHECK:  [3,   5,   7],
  // CHECK:  [7,   9,   11],
  // CHECK:  [11,   13,   15],
  // CHECK:  [15,   17,   19],
  // CHECK:  [19,   21,   23],
  // CHECK:  [23,   25,   27]
  call @exp_xsmm(%A, %B) : (!eltwise, !eltwise) -> ()
  call @printMemrefF32(%B_ud) : (memref<*xf32>) -> ()
  // CHECK:  [-5,   0.0183152,   0.0497804],
  // CHECK:  [-1,   0.0497804,   0.135335],
  // CHECK:  [3,   0.135335,   0.367705],
  // CHECK:  [7,   0.367705,   1],
  // CHECK:  [11,   1,   2.7175],
  // CHECK:  [15,   2.7175,   7.38904],
  // CHECK:  [19,   7.38904,   20.0837],
  // CHECK:  [23,   20.0837,   54.5965]
  call @relu_xsmm(%B, %B) : (!eltwise, !eltwise) -> ()
  call @printMemrefF32(%B_ud) : (memref<*xf32>) -> ()
  // CHECK:  [-5,   0.0183152,   0.0497804],
  // CHECK:  [0,   0.0497804,   0.135335],
  // CHECK:  [3,   0.135335,   0.367705],
  // CHECK:  [7,   0.367705,   1],
  // CHECK:  [11,   1,   2.7175],
  // CHECK:  [15,   2.7175,   7.38904],
  // CHECK:  [19,   7.38904,   20.0837],
  // CHECK:  [23,   20.0837,   54.5965]
  memref.dealloc %B : !eltwise
  memref.dealloc %A : !eltwise
  return
}

func.func @exp_xsmm(%I: !eltwise, %O: !eltwise) {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %exp = xsmm.unary.dispatch EXP(f32, [8, 2], 3, 3, 0) : (f32) -> f32
  xsmm.unary.invoke %O[%c0, %c1] = %exp(%I[%c0, %c1]) : (!eltwise) -> !eltwise
  return
}

func.func @relu_xsmm(%I: !eltwise, %O: !eltwise) {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %relu = xsmm.unary.dispatch RELU(f32, [7, 3], 3, 3, 0) : (f32) -> f32
  xsmm.unary.invoke %O[%c1, %c0] = %relu(%I[%c1, %c0]) : (!eltwise) -> !eltwise
  return
}
