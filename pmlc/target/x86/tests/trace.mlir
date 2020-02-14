// RUN: pmlc-opt -convert-tile-to-pxa -target-cpu %s | FileCheck %s
// -e test_trace -entry-point-result=void

func @test_trace() {
  %c1 = constant 1 : i32
  "tile.trace"(%c1) {msg = "msg"} : (i32) -> (i32)
  return
}

// CHECK-DAG:  llvm.mlir.global internal constant @__trace_msg{{.*}}("msg\00")
// CHECK-DAG:  llvm.func @plaidml_rt_trace(!llvm<"i8*">)
// CHECK:      llvm.func @__trace{{.*}}()
// CHECK:        llvm.mlir.addressof @{{.*}}
// CHECK:        llvm.mlir.constant
// CHECK:        llvm.getelementptr
// CHECK:        llvm.call @plaidml_rt_trace(%{{.*}})
// CHECK:      llvm.func @test_trace
// CHECK:        llvm.call @__trace{{.*}}()
