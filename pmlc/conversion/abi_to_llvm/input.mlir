Converting fn: llvm.func @main(%arg0: !llvm.ptr<float>, %arg1: !llvm.ptr<float>, %arg2: !llvm.i64, %arg3: !llvm.i64, %arg4: !llvm.i64, %arg5: !llvm.i64, %arg6: !llvm.i64, %arg7: !llvm.ptr<float>, %arg8: !llvm.ptr<float>, %arg9: !llvm.i64, %arg10: !llvm.i64, %arg11: !llvm.i64, %arg12: !llvm.i64, %arg13: !llvm.i64, %arg14: !llvm.ptr<float>, %arg15: !llvm.ptr<float>, %arg16: !llvm.i64, %arg17: !llvm.i64, %arg18: !llvm.i64, %arg19: !llvm.i64, %arg20: !llvm.i64) {
  %0 = llvm.mlir.undef : !llvm.struct<(ptr<float>, ptr<float>, i64, array<2 x i64>, array<2 x i64>)>
  %1 = llvm.insertvalue %arg0, %0[0] : !llvm.struct<(ptr<float>, ptr<float>, i64, array<2 x i64>, array<2 x i64>)>
  %2 = llvm.insertvalue %arg1, %1[1] : !llvm.struct<(ptr<float>, ptr<float>, i64, array<2 x i64>, array<2 x i64>)>
  %3 = llvm.insertvalue %arg2, %2[2] : !llvm.struct<(ptr<float>, ptr<float>, i64, array<2 x i64>, array<2 x i64>)>
  %4 = llvm.insertvalue %arg3, %3[3, 0] : !llvm.struct<(ptr<float>, ptr<float>, i64, array<2 x i64>, array<2 x i64>)>
  %5 = llvm.insertvalue %arg5, %4[4, 0] : !llvm.struct<(ptr<float>, ptr<float>, i64, array<2 x i64>, array<2 x i64>)>
  %6 = llvm.insertvalue %arg4, %5[3, 1] : !llvm.struct<(ptr<float>, ptr<float>, i64, array<2 x i64>, array<2 x i64>)>
  %7 = llvm.insertvalue %arg6, %6[4, 1] : !llvm.struct<(ptr<float>, ptr<float>, i64, array<2 x i64>, array<2 x i64>)>
  %8 = llvm.mlir.undef : !llvm.struct<(ptr<float>, ptr<float>, i64, array<2 x i64>, array<2 x i64>)>
  %9 = llvm.insertvalue %arg7, %8[0] : !llvm.struct<(ptr<float>, ptr<float>, i64, array<2 x i64>, array<2 x i64>)>
  %10 = llvm.insertvalue %arg8, %9[1] : !llvm.struct<(ptr<float>, ptr<float>, i64, array<2 x i64>, array<2 x i64>)>
  %11 = llvm.insertvalue %arg9, %10[2] : !llvm.struct<(ptr<float>, ptr<float>, i64, array<2 x i64>, array<2 x i64>)>
  %12 = llvm.insertvalue %arg10, %11[3, 0] : !llvm.struct<(ptr<float>, ptr<float>, i64, array<2 x i64>, array<2 x i64>)>
  %13 = llvm.insertvalue %arg12, %12[4, 0] : !llvm.struct<(ptr<float>, ptr<float>, i64, array<2 x i64>, array<2 x i64>)>
  %14 = llvm.insertvalue %arg11, %13[3, 1] : !llvm.struct<(ptr<float>, ptr<float>, i64, array<2 x i64>, array<2 x i64>)>
  %15 = llvm.insertvalue %arg13, %14[4, 1] : !llvm.struct<(ptr<float>, ptr<float>, i64, array<2 x i64>, array<2 x i64>)>
  %16 = llvm.mlir.undef : !llvm.struct<(ptr<float>, ptr<float>, i64, array<2 x i64>, array<2 x i64>)>
  %17 = llvm.insertvalue %arg14, %16[0] : !llvm.struct<(ptr<float>, ptr<float>, i64, array<2 x i64>, array<2 x i64>)>
  %18 = llvm.insertvalue %arg15, %17[1] : !llvm.struct<(ptr<float>, ptr<float>, i64, array<2 x i64>, array<2 x i64>)>
  %19 = llvm.insertvalue %arg16, %18[2] : !llvm.struct<(ptr<float>, ptr<float>, i64, array<2 x i64>, array<2 x i64>)>
  %20 = llvm.insertvalue %arg17, %19[3, 0] : !llvm.struct<(ptr<float>, ptr<float>, i64, array<2 x i64>, array<2 x i64>)>
  %21 = llvm.insertvalue %arg19, %20[4, 0] : !llvm.struct<(ptr<float>, ptr<float>, i64, array<2 x i64>, array<2 x i64>)>
  %22 = llvm.insertvalue %arg18, %21[3, 1] : !llvm.struct<(ptr<float>, ptr<float>, i64, array<2 x i64>, array<2 x i64>)>
  %23 = llvm.insertvalue %arg20, %22[4, 1] : !llvm.struct<(ptr<float>, ptr<float>, i64, array<2 x i64>, array<2 x i64>)>
  %cst = constant 0.000000e+00 : f32
  %c8 = constant 8 : index
  %c0 = constant 0 : index
  %c32 = constant 32 : index
  %c1 = constant 1 : index
  %24 = xsmm.gemm.dispatch [8, 32, 16], [16, 32, 32]
  "abi.loop"(<<UNKNOWN SSA VALUE>>, <<UNKNOWN SSA VALUE>>, <<UNKNOWN SSA VALUE>>) ( {
  ^bb0(%arg21: memref<8x16xf32>, %arg22: memref<16x32xf32>, %arg23: memref<8x32xf32>):  // no predecessors
    br ^bb1(%c0 : index)
  ^bb1(%25: index):  // 2 preds: ^bb0, ^bb5
    %26 = cmpi "slt", %25, %c8 : index
    cond_br %26, ^bb2, ^bb6
  ^bb2:  // pred: ^bb1
    br ^bb3(%c0 : index)
  ^bb3(%27: index):  // 2 preds: ^bb2, ^bb4
    %28 = cmpi "slt", %27, %c32 : index
    cond_br %28, ^bb4, ^bb5
  ^bb4:  // pred: ^bb3
    store %cst, %arg23[%25, %27] : memref<8x32xf32>
    %29 = addi %27, %c1 : index
    br ^bb3(%29 : index)
  ^bb5:  // pred: ^bb3
    %30 = addi %25, %c1 : index
    br ^bb1(%30 : index)
  ^bb6:  // pred: ^bb1
    xsmm.gemm.invoke %24, %arg23[%c0, %c0] = %arg21[%c0, %c0], %arg22[%c0, %c0] : (memref<8x16xf32>, memref<16x32xf32>) -> memref<8x32xf32>
    "abi.done"() : () -> ()
  }) : (memref<8x16xf32>, memref<16x32xf32>, memref<8x32xf32>) -> ()
  return
}
