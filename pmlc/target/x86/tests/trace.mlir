// RUN: pmlc-opt -convert-tile-to-pxa -target-cpu %s | FileCheck %s
// RUN: pmlc-opt -convert-tile-to-pxa -target-cpu %s | pmlc-jit | FileCheck %s --check-prefix=JIT

func @main() {
  %c1 = constant 1.0 : f32
  "tile.trace"(%c1) {msg = "msg"} : (f32) -> (f32)
  return
}

// CHECK-DAG:  llvm.mlir.global internal constant @__trace_msg{{.*}}("msg\00")
// CHECK-DAG:  llvm.func @plaidml_rt_trace(!llvm.ptr<i8>)
// CHECK:      llvm.func @__trace{{.*}}()
// CHECK:        llvm.mlir.addressof @{{.*}}
// CHECK:        llvm.mlir.constant
// CHECK:        llvm.getelementptr
// CHECK:        llvm.call @plaidml_rt_trace(%{{.*}})
// CHECK:      llvm.func @main
// CHECK:        llvm.call @__trace{{.*}}()

// JIT: msg
