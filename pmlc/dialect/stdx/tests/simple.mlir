// RUN: pmlc-opt %s | pmlc-opt | FileCheck %s

func @std_for(%A: memref<10xf32>) {
  %c0 = constant 0 : index
  %c1 = constant 1 : index
  %c10 = constant 10 : index
  %cst = constant 1.0 : f32
  loop.for %i = %c0 to %c10 step %c1 {
    %0 = load %A[%i] : memref<10xf32>
    %sum = addf %0, %cst : f32
  }
  return
}

// llvm.func @std_for(%arg0: !llvm<"{ float*, float*, i64, [1 x i64], [1 x i64] }*">) {
//   %0 = llvm.load %arg0 : !llvm<"{ float*, float*, i64, [1 x i64], [1 x i64] }*">
//   %1 = llvm.mlir.constant(0 : index) : !llvm.i64
//   %2 = llvm.mlir.constant(1 : index) : !llvm.i64
//   %3 = llvm.mlir.constant(10 : index) : !llvm.i64
//   %4 = llvm.mlir.constant(1.000000e+00 : f32) : !llvm.float
//   llvm.br ^bb1(%1 : !llvm.i64)
// ^bb1(%5: !llvm.i64):	// 2 preds: ^bb0, ^bb2
//   %6 = llvm.icmp "slt" %5, %3 : !llvm.i64
//   llvm.cond_br %6, ^bb2, ^bb3
// ^bb2:	// pred: ^bb1
//   %7 = llvm.extractvalue %0[1] : !llvm<"{ float*, float*, i64, [1 x i64], [1 x i64] }">
//   %8 = llvm.mlir.constant(0 : index) : !llvm.i64
//   %9 = llvm.mlir.constant(1 : index) : !llvm.i64
//   %10 = llvm.mul %5, %9 : !llvm.i64
//   %11 = llvm.add %8, %10 : !llvm.i64
//   %12 = llvm.getelementptr %7[%11] : (!llvm<"float*">, !llvm.i64) -> !llvm<"float*">
//   %13 = llvm.load %12 : !llvm<"float*">
//   %14 = llvm.fadd %13, %4 : !llvm.float
//   %15 = llvm.add %5, %2 : !llvm.i64
//   llvm.br ^bb1(%15 : !llvm.i64)
// ^bb3:	// pred: ^bb1
//   llvm.return
// }
