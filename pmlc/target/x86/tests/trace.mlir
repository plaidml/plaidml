// RUN: pmlc-opt -convert-tile-to-pxa -target-cpu %s | FileCheck %s

module {
  func @test_trace(%arg0: tensor<10x!eltwise.f32>) -> tensor<10x!eltwise.f32> {
    %0 = "tile.trace"(%arg0) {msg = "msg"} : (tensor<10x!eltwise.f32>) -> tensor<10x!eltwise.f32>
    return %0 : tensor<10x!eltwise.f32>
  }
}

// CHECK-DAG:  llvm.mlir.global internal constant @__trace_msg{{.*}}("msg\00")
// CHECK-DAG:  llvm.func @plaidml_rt_trace(!llvm<"i8*">)
// CHECK:      llvm.func @__trace{{.*}}()
// CHECK:        %{{.*}} = llvm.mlir.addressof @{{.*}}
// CHECK:        %{{.*}} = llvm.mlir.constant
// CHECK:        %{{.*}} = llvm.getelementptr
// CHECK:        %{{.*}} = llvm.call @plaidml_rt_trace(%{{.*}})
// CHECK:      llvm.func @test_trace
// CHECK:        llvm.call @__trace{{.*}}()
