// Before lowering to ABI:

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

// After lowering to ABI:

module @dot {
  abi.loop init  {
  ^bb0(%arg0: !llvm.ptr<i8>):  // no predecessors
    abi.yield
  } yield [] body  {
  ^bb0(%arg0: memref<8x16xf32>, %arg1: memref<16x32xf32>, %arg2: memref<8x32xf32>):  // no predecessors
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
    abi.terminator
  } fini  {
    abi.terminator
  }
  func @plaidml_rt_thread_num() -> index
}

// After hoisting and canonicalizing:

module @dot {
  abi.loop init  {
  ^bb0(%arg0: !llvm.ptr<i8>):  // no predecessors
    %0 = xsmm.gemm.dispatch.f32 [8, 32, 16], [16, 32, 32]
    abi.yield %0
  } yield [i64] body  {
  ^bb0(%arg0: i64, %arg1: memref<8x16xf32>, %arg2: memref<16x32xf32>, %arg3: memref<8x32xf32>):  // no predecessors
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
      %0 = call @plaidml_rt_thread_num() : () -> index
      br ^bb1(%c0 : index)
    ^bb1(%1: index):  // 2 preds: ^bb0, ^bb2
      %2 = cmpi "slt", %1, %c8 : index
      cond_br %2, ^bb2, ^bb3
    ^bb2:  // pred: ^bb1
      %3 = muli %0, %c8 : index
      store %cst, %arg3[%1, %3] : memref<8x32xf32>
      %4 = addi %3, %c1 : index
      store %cst, %arg3[%1, %4] : memref<8x32xf32>
      %5 = addi %3, %c2 : index
      store %cst, %arg3[%1, %5] : memref<8x32xf32>
      %6 = addi %3, %c3 : index
      store %cst, %arg3[%1, %6] : memref<8x32xf32>
      %7 = addi %3, %c4 : index
      store %cst, %arg3[%1, %7] : memref<8x32xf32>
      %8 = addi %3, %c5 : index
      store %cst, %arg3[%1, %8] : memref<8x32xf32>
      %9 = addi %3, %c6 : index
      store %cst, %arg3[%1, %9] : memref<8x32xf32>
      %10 = addi %3, %c7 : index
      store %cst, %arg3[%1, %10] : memref<8x32xf32>
      %11 = addi %1, %c1 : index
      br ^bb1(%11 : index)
    ^bb3:  // pred: ^bb1
      omp.terminator
    }
    xsmm.gemm.invoke.f32 %arg0, %arg3[%c0, %c0] = %arg1[%c0, %c0], %arg2[%c0, %c0] : (memref<8x16xf32>, memref<16x32xf32>) -> memref<8x32xf32>
    abi.terminator
  } fini  {
  ^bb0(%arg0: i64):  // no predecessors
    abi.terminator
  }
  func @plaidml_rt_thread_num() -> index
}

// After conversion to LLVMIR:

module @dot {
  llvm.func @plaidml_rt_xsmm_gemm_invoke_f32(!llvm.i64, !llvm.ptr<float>, !llvm.ptr<float>, !llvm.ptr<float>)
  llvm.func @malloc(!llvm.i64) -> !llvm.ptr<i8>
  llvm.func @plaidml_rt_xsmm_gemm_dispatch_f32(!llvm.i32, !llvm.i32, !llvm.i32, !llvm.i32, !llvm.i32, !llvm.i32) -> !llvm.i64
  llvm.func @free(!llvm.ptr<i8>)
  llvm.func @plaidml_init(%arg0: !llvm.ptr<i8>, %arg1: !llvm.ptr<i8>) -> !llvm.ptr<struct<(i64)>> {
    %0 = llvm.mlir.constant(16 : i64) : !llvm.i32
    %1 = llvm.mlir.constant(32 : i64) : !llvm.i32
    %2 = llvm.mlir.constant(32 : i64) : !llvm.i32
    %3 = llvm.mlir.constant(8 : i64) : !llvm.i32
    %4 = llvm.mlir.constant(32 : i64) : !llvm.i32
    %5 = llvm.mlir.constant(16 : i64) : !llvm.i32
    %6 = llvm.call @plaidml_rt_xsmm_gemm_dispatch_f32(%0, %1, %2, %3, %4, %5) : (!llvm.i32, !llvm.i32, !llvm.i32, !llvm.i32, !llvm.i32, !llvm.i32) -> !llvm.i64
    %7 = llvm.mlir.undef : !llvm.struct<(i64)>
    %8 = llvm.insertvalue %6, %7[0] : !llvm.struct<(i64)>
    %9 = llvm.mlir.null : !llvm.ptr<struct<(i64)>>
    %10 = llvm.mlir.constant(1 : index) : !llvm.i64
    %11 = llvm.getelementptr %9[%10] : (!llvm.ptr<struct<(i64)>>, !llvm.i64) -> !llvm.ptr<struct<(i64)>>
    %12 = llvm.ptrtoint %11 : !llvm.ptr<struct<(i64)>> to !llvm.i64
    %13 = llvm.call @malloc(%12) : (!llvm.i64) -> !llvm.ptr<i8>
    %14 = llvm.bitcast %13 : !llvm.ptr<i8> to !llvm.ptr<struct<(i64)>>
    llvm.store %8, %14 : !llvm.ptr<struct<(i64)>>
    llvm.return %14 : !llvm.ptr<struct<(i64)>>
  }
  llvm.func @plaidml_exec(%arg0: !llvm.ptr<struct<(i64)>>, %arg1: !llvm.ptr<struct<(ptr<struct<(ptr<float>, ptr<float>, i64, array<2 x i64>, array<2 x i64>)>>, ptr<struct<(ptr<float>, ptr<float>, i64, array<2 x i64>, array<2 x i64>)>>, ptr<struct<(ptr<float>, ptr<float>, i64, array<2 x i64>, array<2 x i64>)>>)>>) {
    %0 = llvm.load %arg0 : !llvm.ptr<struct<(i64)>>
    %1 = llvm.extractvalue %0[0] : !llvm.struct<(i64)>
    %2 = llvm.load %arg1 : !llvm.ptr<struct<(ptr<struct<(ptr<float>, ptr<float>, i64, array<2 x i64>, array<2 x i64>)>>, ptr<struct<(ptr<float>, ptr<float>, i64, array<2 x i64>, array<2 x i64>)>>, ptr<struct<(ptr<float>, ptr<float>, i64, array<2 x i64>, array<2 x i64>)>>)>>
    %3 = llvm.extractvalue %2[0] : !llvm.struct<(ptr<struct<(ptr<float>, ptr<float>, i64, array<2 x i64>, array<2 x i64>)>>, ptr<struct<(ptr<float>, ptr<float>, i64, array<2 x i64>, array<2 x i64>)>>, ptr<struct<(ptr<float>, ptr<float>, i64, array<2 x i64>, array<2 x i64>)>>)>
    %4 = llvm.load %3 : !llvm.ptr<struct<(ptr<float>, ptr<float>, i64, array<2 x i64>, array<2 x i64>)>>
    %5 = llvm.extractvalue %2[1] : !llvm.struct<(ptr<struct<(ptr<float>, ptr<float>, i64, array<2 x i64>, array<2 x i64>)>>, ptr<struct<(ptr<float>, ptr<float>, i64, array<2 x i64>, array<2 x i64>)>>, ptr<struct<(ptr<float>, ptr<float>, i64, array<2 x i64>, array<2 x i64>)>>)>
    %6 = llvm.load %5 : !llvm.ptr<struct<(ptr<float>, ptr<float>, i64, array<2 x i64>, array<2 x i64>)>>
    %7 = llvm.extractvalue %2[2] : !llvm.struct<(ptr<struct<(ptr<float>, ptr<float>, i64, array<2 x i64>, array<2 x i64>)>>, ptr<struct<(ptr<float>, ptr<float>, i64, array<2 x i64>, array<2 x i64>)>>, ptr<struct<(ptr<float>, ptr<float>, i64, array<2 x i64>, array<2 x i64>)>>)>
    %8 = llvm.load %7 : !llvm.ptr<struct<(ptr<float>, ptr<float>, i64, array<2 x i64>, array<2 x i64>)>>
    %9 = llvm.mlir.constant(0.000000e+00 : f32) : !llvm.float
    %10 = llvm.mlir.constant(0 : index) : !llvm.i64
    %11 = llvm.mlir.constant(1 : index) : !llvm.i64
    %12 = llvm.mlir.constant(2 : index) : !llvm.i64
    %13 = llvm.mlir.constant(3 : index) : !llvm.i64
    %14 = llvm.mlir.constant(5 : index) : !llvm.i64
    %15 = llvm.mlir.constant(6 : index) : !llvm.i64
    %16 = llvm.mlir.constant(8 : index) : !llvm.i64
    %17 = llvm.mlir.constant(7 : index) : !llvm.i64
    %18 = llvm.mlir.constant(4 : index) : !llvm.i64
    omp.parallel num_threads(%18 : !llvm.i64) default(shared) {
      %46 = llvm.call @plaidml_rt_thread_num() : () -> !llvm.i64
      llvm.br ^bb1(%10 : !llvm.i64)
    ^bb1(%47: !llvm.i64):  // 2 preds: ^bb0, ^bb2
      %48 = llvm.icmp "slt" %47, %16 : !llvm.i64
      llvm.cond_br %48, ^bb2, ^bb3
    ^bb2:  // pred: ^bb1
      %49 = llvm.mul %46, %16 : !llvm.i64
      %50 = llvm.extractvalue %8[1] : !llvm.struct<(ptr<float>, ptr<float>, i64, array<2 x i64>, array<2 x i64>)>
      %51 = llvm.mlir.constant(0 : index) : !llvm.i64
      %52 = llvm.mlir.constant(32 : index) : !llvm.i64
      %53 = llvm.mul %47, %52 : !llvm.i64
      %54 = llvm.add %51, %53 : !llvm.i64
      %55 = llvm.mlir.constant(1 : index) : !llvm.i64
      %56 = llvm.mul %49, %55 : !llvm.i64
      %57 = llvm.add %54, %56 : !llvm.i64
      %58 = llvm.getelementptr %50[%57] : (!llvm.ptr<float>, !llvm.i64) -> !llvm.ptr<float>
      llvm.store %9, %58 : !llvm.ptr<float>
      %59 = llvm.add %49, %11 : !llvm.i64
      %60 = llvm.extractvalue %8[1] : !llvm.struct<(ptr<float>, ptr<float>, i64, array<2 x i64>, array<2 x i64>)>
      %61 = llvm.mlir.constant(0 : index) : !llvm.i64
      %62 = llvm.mlir.constant(32 : index) : !llvm.i64
      %63 = llvm.mul %47, %62 : !llvm.i64
      %64 = llvm.add %61, %63 : !llvm.i64
      %65 = llvm.mlir.constant(1 : index) : !llvm.i64
      %66 = llvm.mul %59, %65 : !llvm.i64
      %67 = llvm.add %64, %66 : !llvm.i64
      %68 = llvm.getelementptr %60[%67] : (!llvm.ptr<float>, !llvm.i64) -> !llvm.ptr<float>
      llvm.store %9, %68 : !llvm.ptr<float>
      %69 = llvm.add %49, %12 : !llvm.i64
      %70 = llvm.extractvalue %8[1] : !llvm.struct<(ptr<float>, ptr<float>, i64, array<2 x i64>, array<2 x i64>)>
      %71 = llvm.mlir.constant(0 : index) : !llvm.i64
      %72 = llvm.mlir.constant(32 : index) : !llvm.i64
      %73 = llvm.mul %47, %72 : !llvm.i64
      %74 = llvm.add %71, %73 : !llvm.i64
      %75 = llvm.mlir.constant(1 : index) : !llvm.i64
      %76 = llvm.mul %69, %75 : !llvm.i64
      %77 = llvm.add %74, %76 : !llvm.i64
      %78 = llvm.getelementptr %70[%77] : (!llvm.ptr<float>, !llvm.i64) -> !llvm.ptr<float>
      llvm.store %9, %78 : !llvm.ptr<float>
      %79 = llvm.add %49, %13 : !llvm.i64
      %80 = llvm.extractvalue %8[1] : !llvm.struct<(ptr<float>, ptr<float>, i64, array<2 x i64>, array<2 x i64>)>
      %81 = llvm.mlir.constant(0 : index) : !llvm.i64
      %82 = llvm.mlir.constant(32 : index) : !llvm.i64
      %83 = llvm.mul %47, %82 : !llvm.i64
      %84 = llvm.add %81, %83 : !llvm.i64
      %85 = llvm.mlir.constant(1 : index) : !llvm.i64
      %86 = llvm.mul %79, %85 : !llvm.i64
      %87 = llvm.add %84, %86 : !llvm.i64
      %88 = llvm.getelementptr %80[%87] : (!llvm.ptr<float>, !llvm.i64) -> !llvm.ptr<float>
      llvm.store %9, %88 : !llvm.ptr<float>
      %89 = llvm.add %49, %18 : !llvm.i64
      %90 = llvm.extractvalue %8[1] : !llvm.struct<(ptr<float>, ptr<float>, i64, array<2 x i64>, array<2 x i64>)>
      %91 = llvm.mlir.constant(0 : index) : !llvm.i64
      %92 = llvm.mlir.constant(32 : index) : !llvm.i64
      %93 = llvm.mul %47, %92 : !llvm.i64
      %94 = llvm.add %91, %93 : !llvm.i64
      %95 = llvm.mlir.constant(1 : index) : !llvm.i64
      %96 = llvm.mul %89, %95 : !llvm.i64
      %97 = llvm.add %94, %96 : !llvm.i64
      %98 = llvm.getelementptr %90[%97] : (!llvm.ptr<float>, !llvm.i64) -> !llvm.ptr<float>
      llvm.store %9, %98 : !llvm.ptr<float>
      %99 = llvm.add %49, %14 : !llvm.i64
      %100 = llvm.extractvalue %8[1] : !llvm.struct<(ptr<float>, ptr<float>, i64, array<2 x i64>, array<2 x i64>)>
      %101 = llvm.mlir.constant(0 : index) : !llvm.i64
      %102 = llvm.mlir.constant(32 : index) : !llvm.i64
      %103 = llvm.mul %47, %102 : !llvm.i64
      %104 = llvm.add %101, %103 : !llvm.i64
      %105 = llvm.mlir.constant(1 : index) : !llvm.i64
      %106 = llvm.mul %99, %105 : !llvm.i64
      %107 = llvm.add %104, %106 : !llvm.i64
      %108 = llvm.getelementptr %100[%107] : (!llvm.ptr<float>, !llvm.i64) -> !llvm.ptr<float>
      llvm.store %9, %108 : !llvm.ptr<float>
      %109 = llvm.add %49, %15 : !llvm.i64
      %110 = llvm.extractvalue %8[1] : !llvm.struct<(ptr<float>, ptr<float>, i64, array<2 x i64>, array<2 x i64>)>
      %111 = llvm.mlir.constant(0 : index) : !llvm.i64
      %112 = llvm.mlir.constant(32 : index) : !llvm.i64
      %113 = llvm.mul %47, %112 : !llvm.i64
      %114 = llvm.add %111, %113 : !llvm.i64
      %115 = llvm.mlir.constant(1 : index) : !llvm.i64
      %116 = llvm.mul %109, %115 : !llvm.i64
      %117 = llvm.add %114, %116 : !llvm.i64
      %118 = llvm.getelementptr %110[%117] : (!llvm.ptr<float>, !llvm.i64) -> !llvm.ptr<float>
      llvm.store %9, %118 : !llvm.ptr<float>
      %119 = llvm.add %49, %17 : !llvm.i64
      %120 = llvm.extractvalue %8[1] : !llvm.struct<(ptr<float>, ptr<float>, i64, array<2 x i64>, array<2 x i64>)>
      %121 = llvm.mlir.constant(0 : index) : !llvm.i64
      %122 = llvm.mlir.constant(32 : index) : !llvm.i64
      %123 = llvm.mul %47, %122 : !llvm.i64
      %124 = llvm.add %121, %123 : !llvm.i64
      %125 = llvm.mlir.constant(1 : index) : !llvm.i64
      %126 = llvm.mul %119, %125 : !llvm.i64
      %127 = llvm.add %124, %126 : !llvm.i64
      %128 = llvm.getelementptr %120[%127] : (!llvm.ptr<float>, !llvm.i64) -> !llvm.ptr<float>
      llvm.store %9, %128 : !llvm.ptr<float>
      %129 = llvm.add %47, %11 : !llvm.i64
      llvm.br ^bb1(%129 : !llvm.i64)
    ^bb3:  // pred: ^bb1
      omp.terminator
    }
    %19 = llvm.extractvalue %4[1] : !llvm.struct<(ptr<float>, ptr<float>, i64, array<2 x i64>, array<2 x i64>)>
    %20 = llvm.mlir.constant(0 : index) : !llvm.i64
    %21 = llvm.mlir.constant(16 : index) : !llvm.i64
    %22 = llvm.mul %10, %21 : !llvm.i64
    %23 = llvm.add %20, %22 : !llvm.i64
    %24 = llvm.mlir.constant(1 : index) : !llvm.i64
    %25 = llvm.mul %10, %24 : !llvm.i64
    %26 = llvm.add %23, %25 : !llvm.i64
    %27 = llvm.getelementptr %19[%26] : (!llvm.ptr<float>, !llvm.i64) -> !llvm.ptr<float>
    %28 = llvm.extractvalue %6[1] : !llvm.struct<(ptr<float>, ptr<float>, i64, array<2 x i64>, array<2 x i64>)>
    %29 = llvm.mlir.constant(0 : index) : !llvm.i64
    %30 = llvm.mlir.constant(32 : index) : !llvm.i64
    %31 = llvm.mul %10, %30 : !llvm.i64
    %32 = llvm.add %29, %31 : !llvm.i64
    %33 = llvm.mlir.constant(1 : index) : !llvm.i64
    %34 = llvm.mul %10, %33 : !llvm.i64
    %35 = llvm.add %32, %34 : !llvm.i64
    %36 = llvm.getelementptr %28[%35] : (!llvm.ptr<float>, !llvm.i64) -> !llvm.ptr<float>
    %37 = llvm.extractvalue %8[1] : !llvm.struct<(ptr<float>, ptr<float>, i64, array<2 x i64>, array<2 x i64>)>
    %38 = llvm.mlir.constant(0 : index) : !llvm.i64
    %39 = llvm.mlir.constant(32 : index) : !llvm.i64
    %40 = llvm.mul %10, %39 : !llvm.i64
    %41 = llvm.add %38, %40 : !llvm.i64
    %42 = llvm.mlir.constant(1 : index) : !llvm.i64
    %43 = llvm.mul %10, %42 : !llvm.i64
    %44 = llvm.add %41, %43 : !llvm.i64
    %45 = llvm.getelementptr %37[%44] : (!llvm.ptr<float>, !llvm.i64) -> !llvm.ptr<float>
    llvm.call @plaidml_rt_xsmm_gemm_invoke_f32(%1, %27, %36, %45) : (!llvm.i64, !llvm.ptr<float>, !llvm.ptr<float>, !llvm.ptr<float>) -> ()
    llvm.return
  }
  llvm.func @plaidml_fini(%arg0: !llvm.ptr<struct<(i64)>>) {
    %0 = llvm.load %arg0 : !llvm.ptr<struct<(i64)>>
    %1 = llvm.extractvalue %0[0] : !llvm.struct<(i64)>
    %2 = llvm.bitcast %arg0 : !llvm.ptr<struct<(i64)>> to !llvm.ptr<i8>
    llvm.call @free(%2) : (!llvm.ptr<i8>) -> ()
    llvm.return
  }
  llvm.func @plaidml_rt_thread_num() -> !llvm.i64 {
    %0 = llvm.call @_mlir_ciface_plaidml_rt_thread_num() : () -> !llvm.i64
    llvm.return %0 : !llvm.i64
  }
  llvm.func @_mlir_ciface_plaidml_rt_thread_num() -> !llvm.i64
}
