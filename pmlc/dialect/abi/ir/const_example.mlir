// Before lowering to ABI:

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

// After lowering to ABI:

module @const_add {
  abi.loop init  {
  ^bb0(%arg0: !llvm.ptr<i8>, %arg1: memref<4xi32>, %arg2: memref<4xi32>):  // no predecessors
    abi.yield %arg1, %arg2 : (memref<4xi32>, memref<4xi32>)
  } yield [memref<4xi32>, memref<4xi32>] body  {
  ^bb0(%arg0: memref<4xi32>, %arg1: memref<4xi32>, %arg2: memref<4xi32>):  // no predecessors
    %c4 = constant 4 : index
    omp.parallel num_threads(%c4 : index) default(shared) {
      %0 = call @plaidml_rt_thread_num() : () -> index
      %1 = load %arg0[%0] : memref<4xi32>
      %2 = load %arg1[%0] : memref<4xi32>
      %3 = addi %1, %2 : i32
      store %3, %arg2[%0] : memref<4xi32>
      omp.terminator
    }
    abi.terminator
  } fini  {
  ^bb0(%arg0: memref<4xi32>, %arg1: memref<4xi32>):  // no predecessors
    abi.terminator
  }
  func @plaidml_rt_thread_num() -> index
}

// After hoisting and canonicalizing:

module @const_add {
  abi.loop init  {
  ^bb0(%arg0: !llvm.ptr<i8>, %arg1: memref<4xi32>, %arg2: memref<4xi32>):  // no predecessors
    abi.yield %arg1, %arg2 : (memref<4xi32>, memref<4xi32>)
  } yield [memref<4xi32>, memref<4xi32>] body  {
  ^bb0(%arg0: memref<4xi32>, %arg1: memref<4xi32>, %arg2: memref<4xi32>):  // no predecessors
    %c4 = constant 4 : index
    omp.parallel num_threads(%c4 : index) default(shared) {
      %0 = call @plaidml_rt_thread_num() : () -> index
      %1 = load %arg0[%0] : memref<4xi32>
      %2 = load %arg1[%0] : memref<4xi32>
      %3 = addi %1, %2 : i32
      store %3, %arg2[%0] : memref<4xi32>
      omp.terminator
    }
    abi.terminator
  } fini  {
  ^bb0(%arg0: memref<4xi32>, %arg1: memref<4xi32>):  // no predecessors
    abi.terminator
  }
  func @plaidml_rt_thread_num() -> index
}

// After conversion to LLVMIR:


module @const_add {
  llvm.func @malloc(!llvm.i64) -> !llvm.ptr<i8>
  llvm.func @free(!llvm.ptr<i8>)
  llvm.func @plaidml_init(%arg0: !llvm.ptr<i8>, %arg1: !llvm.ptr<struct<(ptr<struct<(ptr<i32>, ptr<i32>, i64, array<1 x i64>, array<1 x i64>)>>, ptr<struct<(ptr<i32>, ptr<i32>, i64, array<1 x i64>, array<1 x i64>)>>)>>) -> !llvm.ptr<struct<(struct<(ptr<i32>, ptr<i32>, i64, array<1 x i64>, array<1 x i64>)>, struct<(ptr<i32>, ptr<i32>, i64, array<1 x i64>, array<1 x i64>)>)>> {
    %0 = llvm.load %arg1 : !llvm.ptr<struct<(ptr<struct<(ptr<i32>, ptr<i32>, i64, array<1 x i64>, array<1 x i64>)>>, ptr<struct<(ptr<i32>, ptr<i32>, i64, array<1 x i64>, array<1 x i64>)>>)>>
    %1 = llvm.extractvalue %0[0] : !llvm.struct<(ptr<struct<(ptr<i32>, ptr<i32>, i64, array<1 x i64>, array<1 x i64>)>>, ptr<struct<(ptr<i32>, ptr<i32>, i64, array<1 x i64>, array<1 x i64>)>>)>
    %2 = llvm.load %1 : !llvm.ptr<struct<(ptr<i32>, ptr<i32>, i64, array<1 x i64>, array<1 x i64>)>>
    %3 = llvm.extractvalue %0[1] : !llvm.struct<(ptr<struct<(ptr<i32>, ptr<i32>, i64, array<1 x i64>, array<1 x i64>)>>, ptr<struct<(ptr<i32>, ptr<i32>, i64, array<1 x i64>, array<1 x i64>)>>)>
    %4 = llvm.load %3 : !llvm.ptr<struct<(ptr<i32>, ptr<i32>, i64, array<1 x i64>, array<1 x i64>)>>
    %5 = llvm.mlir.undef : !llvm.struct<(struct<(ptr<i32>, ptr<i32>, i64, array<1 x i64>, array<1 x i64>)>, struct<(ptr<i32>, ptr<i32>, i64, array<1 x i64>, array<1 x i64>)>)>
    %6 = llvm.insertvalue %2, %5[0] : !llvm.struct<(struct<(ptr<i32>, ptr<i32>, i64, array<1 x i64>, array<1 x i64>)>, struct<(ptr<i32>, ptr<i32>, i64, array<1 x i64>, array<1 x i64>)>)>
    %7 = llvm.insertvalue %4, %6[1] : !llvm.struct<(struct<(ptr<i32>, ptr<i32>, i64, array<1 x i64>, array<1 x i64>)>, struct<(ptr<i32>, ptr<i32>, i64, array<1 x i64>, array<1 x i64>)>)>
    %8 = llvm.mlir.null : !llvm.ptr<struct<(struct<(ptr<i32>, ptr<i32>, i64, array<1 x i64>, array<1 x i64>)>, struct<(ptr<i32>, ptr<i32>, i64, array<1 x i64>, array<1 x i64>)>)>>
    %9 = llvm.mlir.constant(1 : index) : !llvm.i64
    %10 = llvm.getelementptr %8[%9] : (!llvm.ptr<struct<(struct<(ptr<i32>, ptr<i32>, i64, array<1 x i64>, array<1 x i64>)>, struct<(ptr<i32>, ptr<i32>, i64, array<1 x i64>, array<1 x i64>)>)>>, !llvm.i64) -> !llvm.ptr<struct<(struct<(ptr<i32>, ptr<i32>, i64, array<1 x i64>, array<1 x i64>)>, struct<(ptr<i32>, ptr<i32>, i64, array<1 x i64>, array<1 x i64>)>)>>
    %11 = llvm.ptrtoint %10 : !llvm.ptr<struct<(struct<(ptr<i32>, ptr<i32>, i64, array<1 x i64>, array<1 x i64>)>, struct<(ptr<i32>, ptr<i32>, i64, array<1 x i64>, array<1 x i64>)>)>> to !llvm.i64
    %12 = llvm.call @malloc(%11) : (!llvm.i64) -> !llvm.ptr<i8>
    %13 = llvm.bitcast %12 : !llvm.ptr<i8> to !llvm.ptr<struct<(struct<(ptr<i32>, ptr<i32>, i64, array<1 x i64>, array<1 x i64>)>, struct<(ptr<i32>, ptr<i32>, i64, array<1 x i64>, array<1 x i64>)>)>>
    llvm.store %7, %13 : !llvm.ptr<struct<(struct<(ptr<i32>, ptr<i32>, i64, array<1 x i64>, array<1 x i64>)>, struct<(ptr<i32>, ptr<i32>, i64, array<1 x i64>, array<1 x i64>)>)>>
    llvm.return %13 : !llvm.ptr<struct<(struct<(ptr<i32>, ptr<i32>, i64, array<1 x i64>, array<1 x i64>)>, struct<(ptr<i32>, ptr<i32>, i64, array<1 x i64>, array<1 x i64>)>)>>
  }
  llvm.func @plaidml_exec(%arg0: !llvm.ptr<struct<(struct<(ptr<i32>, ptr<i32>, i64, array<1 x i64>, array<1 x i64>)>, struct<(ptr<i32>, ptr<i32>, i64, array<1 x i64>, array<1 x i64>)>)>>, %arg1: !llvm.ptr<struct<(ptr<struct<(ptr<i32>, ptr<i32>, i64, array<1 x i64>, array<1 x i64>)>>)>>) {
    %0 = llvm.load %arg0 : !llvm.ptr<struct<(struct<(ptr<i32>, ptr<i32>, i64, array<1 x i64>, array<1 x i64>)>, struct<(ptr<i32>, ptr<i32>, i64, array<1 x i64>, array<1 x i64>)>)>>
    %1 = llvm.extractvalue %0[0] : !llvm.struct<(struct<(ptr<i32>, ptr<i32>, i64, array<1 x i64>, array<1 x i64>)>, struct<(ptr<i32>, ptr<i32>, i64, array<1 x i64>, array<1 x i64>)>)>
    %2 = llvm.extractvalue %0[1] : !llvm.struct<(struct<(ptr<i32>, ptr<i32>, i64, array<1 x i64>, array<1 x i64>)>, struct<(ptr<i32>, ptr<i32>, i64, array<1 x i64>, array<1 x i64>)>)>
    %3 = llvm.load %arg1 : !llvm.ptr<struct<(ptr<struct<(ptr<i32>, ptr<i32>, i64, array<1 x i64>, array<1 x i64>)>>)>>
    %4 = llvm.extractvalue %3[0] : !llvm.struct<(ptr<struct<(ptr<i32>, ptr<i32>, i64, array<1 x i64>, array<1 x i64>)>>)>
    %5 = llvm.load %4 : !llvm.ptr<struct<(ptr<i32>, ptr<i32>, i64, array<1 x i64>, array<1 x i64>)>>
    %6 = llvm.mlir.constant(4 : index) : !llvm.i64
    omp.parallel num_threads(%6 : !llvm.i64) default(shared) {
      %7 = llvm.call @plaidml_rt_thread_num() : () -> !llvm.i64
      %8 = llvm.extractvalue %1[1] : !llvm.struct<(ptr<i32>, ptr<i32>, i64, array<1 x i64>, array<1 x i64>)>
      %9 = llvm.mlir.constant(0 : index) : !llvm.i64
      %10 = llvm.mlir.constant(1 : index) : !llvm.i64
      %11 = llvm.mul %7, %10 : !llvm.i64
      %12 = llvm.add %9, %11 : !llvm.i64
      %13 = llvm.getelementptr %8[%12] : (!llvm.ptr<i32>, !llvm.i64) -> !llvm.ptr<i32>
      %14 = llvm.load %13 : !llvm.ptr<i32>
      %15 = llvm.extractvalue %2[1] : !llvm.struct<(ptr<i32>, ptr<i32>, i64, array<1 x i64>, array<1 x i64>)>
      %16 = llvm.mlir.constant(0 : index) : !llvm.i64
      %17 = llvm.mlir.constant(1 : index) : !llvm.i64
      %18 = llvm.mul %7, %17 : !llvm.i64
      %19 = llvm.add %16, %18 : !llvm.i64
      %20 = llvm.getelementptr %15[%19] : (!llvm.ptr<i32>, !llvm.i64) -> !llvm.ptr<i32>
      %21 = llvm.load %20 : !llvm.ptr<i32>
      %22 = llvm.add %14, %21 : !llvm.i32
      %23 = llvm.extractvalue %5[1] : !llvm.struct<(ptr<i32>, ptr<i32>, i64, array<1 x i64>, array<1 x i64>)>
      %24 = llvm.mlir.constant(0 : index) : !llvm.i64
      %25 = llvm.mlir.constant(1 : index) : !llvm.i64
      %26 = llvm.mul %7, %25 : !llvm.i64
      %27 = llvm.add %24, %26 : !llvm.i64
      %28 = llvm.getelementptr %23[%27] : (!llvm.ptr<i32>, !llvm.i64) -> !llvm.ptr<i32>
      llvm.store %22, %28 : !llvm.ptr<i32>
      omp.terminator
    }
    llvm.return
  }
  llvm.func @plaidml_fini(%arg0: !llvm.ptr<struct<(struct<(ptr<i32>, ptr<i32>, i64, array<1 x i64>, array<1 x i64>)>, struct<(ptr<i32>, ptr<i32>, i64, array<1 x i64>, array<1 x i64>)>)>>) {
    %0 = llvm.load %arg0 : !llvm.ptr<struct<(struct<(ptr<i32>, ptr<i32>, i64, array<1 x i64>, array<1 x i64>)>, struct<(ptr<i32>, ptr<i32>, i64, array<1 x i64>, array<1 x i64>)>)>>
    %1 = llvm.extractvalue %0[0] : !llvm.struct<(struct<(ptr<i32>, ptr<i32>, i64, array<1 x i64>, array<1 x i64>)>, struct<(ptr<i32>, ptr<i32>, i64, array<1 x i64>, array<1 x i64>)>)>
    %2 = llvm.extractvalue %0[1] : !llvm.struct<(struct<(ptr<i32>, ptr<i32>, i64, array<1 x i64>, array<1 x i64>)>, struct<(ptr<i32>, ptr<i32>, i64, array<1 x i64>, array<1 x i64>)>)>
    %3 = llvm.bitcast %arg0 : !llvm.ptr<struct<(struct<(ptr<i32>, ptr<i32>, i64, array<1 x i64>, array<1 x i64>)>, struct<(ptr<i32>, ptr<i32>, i64, array<1 x i64>, array<1 x i64>)>)>> to !llvm.ptr<i8>
    llvm.call @free(%3) : (!llvm.ptr<i8>) -> ()
    llvm.return
  }
  llvm.func @plaidml_rt_thread_num() -> !llvm.i64 {
    %0 = llvm.call @_mlir_ciface_plaidml_rt_thread_num() : () -> !llvm.i64
    llvm.return %0 : !llvm.i64
  }
  llvm.func @_mlir_ciface_plaidml_rt_thread_num() -> !llvm.i64
}
