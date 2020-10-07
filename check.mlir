

module attributes {llvm.data_layout = ""} {
  llvm.func @free(!llvm.ptr<i8>)
  llvm.func @malloc(!llvm.i64) -> !llvm.ptr<i8>
  llvm.func @plaidml_rt_bounds_check(!llvm.i64, !llvm.i64)
  llvm.func @main() {
    %0 = llvm.mlir.constant(0 : index) : !llvm.i64
    %1 = llvm.mlir.constant(10 : index) : !llvm.i64
    %2 = llvm.mlir.constant(20 : index) : !llvm.i64
    %3 = llvm.mlir.constant(10 : index) : !llvm.i64
    %4 = llvm.mul %2, %3 : !llvm.i64
    %5 = llvm.mlir.null : !llvm.ptr<float>
    %6 = llvm.mlir.constant(1 : index) : !llvm.i64
    %7 = llvm.getelementptr %5[%6] : (!llvm.ptr<float>, !llvm.i64) -> !llvm.ptr<float>
    %8 = llvm.ptrtoint %7 : !llvm.ptr<float> to !llvm.i64
    %9 = llvm.mul %4, %8 : !llvm.i64
    %10 = llvm.mlir.constant(1 : index) : !llvm.i64
    %11 = llvm.call @malloc(%9) : (!llvm.i64) -> !llvm.ptr<i8>
    %12 = llvm.bitcast %11 : !llvm.ptr<i8> to !llvm.ptr<float>
    %13 = llvm.mlir.undef : !llvm.struct<(ptr<float>, ptr<float>, i64, array<2 x i64>, array<2 x i64>)>
    %14 = llvm.insertvalue %12, %13[0] : !llvm.struct<(ptr<float>, ptr<float>, i64, array<2 x i64>, array<2 x i64>)>
    %15 = llvm.insertvalue %12, %14[1] : !llvm.struct<(ptr<float>, ptr<float>, i64, array<2 x i64>, array<2 x i64>)>
    %16 = llvm.mlir.constant(0 : index) : !llvm.i64
    %17 = llvm.insertvalue %16, %15[2] : !llvm.struct<(ptr<float>, ptr<float>, i64, array<2 x i64>, array<2 x i64>)>
    %18 = llvm.mlir.constant(1 : index) : !llvm.i64
    %19 = llvm.mlir.constant(10 : index) : !llvm.i64
    %20 = llvm.insertvalue %2, %17[3, 0] : !llvm.struct<(ptr<float>, ptr<float>, i64, array<2 x i64>, array<2 x i64>)>
    %21 = llvm.insertvalue %19, %20[4, 0] : !llvm.struct<(ptr<float>, ptr<float>, i64, array<2 x i64>, array<2 x i64>)>
    %22 = llvm.insertvalue %3, %21[3, 1] : !llvm.struct<(ptr<float>, ptr<float>, i64, array<2 x i64>, array<2 x i64>)>
    %23 = llvm.insertvalue %18, %22[4, 1] : !llvm.struct<(ptr<float>, ptr<float>, i64, array<2 x i64>, array<2 x i64>)>
    %24 = llvm.mlir.constant(20 : i64) : !llvm.i64
    llvm.call @plaidml_rt_bounds_check(%0, %24) : (!llvm.i64, !llvm.i64) -> ()
    %25 = llvm.mlir.constant(10 : i64) : !llvm.i64
    llvm.call @plaidml_rt_bounds_check(%1, %25) : (!llvm.i64, !llvm.i64) -> ()
    %26 = llvm.extractvalue %23[1] : !llvm.struct<(ptr<float>, ptr<float>, i64, array<2 x i64>, array<2 x i64>)>
    %27 = llvm.mlir.constant(0 : index) : !llvm.i64
    %28 = llvm.mlir.constant(10 : index) : !llvm.i64
    %29 = llvm.mul %0, %28 : !llvm.i64
    %30 = llvm.add %27, %29 : !llvm.i64
    %31 = llvm.mlir.constant(1 : index) : !llvm.i64
    %32 = llvm.mul %1, %31 : !llvm.i64
    %33 = llvm.add %30, %32 : !llvm.i64
    %34 = llvm.getelementptr %26[%33] : (!llvm.ptr<float>, !llvm.i64) -> !llvm.ptr<float>
    %35 = llvm.load %34 : !llvm.ptr<float>
    %36 = llvm.extractvalue %23[0] : !llvm.struct<(ptr<float>, ptr<float>, i64, array<2 x i64>, array<2 x i64>)>
    %37 = llvm.bitcast %36 : !llvm.ptr<float> to !llvm.ptr<i8>
    llvm.call @free(%37) : (!llvm.ptr<i8>) -> ()
    llvm.return
  }
  llvm.func @setup(%arg0: !llvm.ptr<i8>) -> !llvm.ptr<i8> {
    llvm.return %arg0 : !llvm.ptr<i8>
  }
  llvm.func @execute(%arg0: !llvm.ptr<i8>) {
    llvm.call @main() : () -> ()
    llvm.return
  }
  llvm.func @teardown(%arg0: !llvm.ptr<i8>) {
    llvm.return
  }
}
