// RUN: pmlc-opt %s -pml-load-module=path=%S/schedule.mlir | FileCheck %s

// CHECK: module @schedule attributes {pml.rules = [{{.*}}]}
// CHECK: func @test()
func @test() {
  return
}
