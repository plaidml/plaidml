// RUN: pmlc-opt -convert-scf-to-omp -canonicalize -cse %s | FileCheck %s

module {
  // CHECK-LABEL: func @test
  func @test(%buffer: memref<100xf32>) {
    %lb = constant 0 : index
    %ub = constant 8 : index
    %step = constant 1 : index
    %init = constant 0.0 : f32
    scf.parallel (%iv) = (%lb) to (%ub) step (%step) init (%init) -> f32 {
      %elem_to_reduce = load %buffer[%iv] : memref<100xf32>
      scf.reduce(%elem_to_reduce) : f32 {
        ^bb0(%lhs : f32, %rhs: f32):
          %res = addf %lhs, %rhs : f32
          scf.reduce.return %res : f32
      }
    } {tags = {cpuBlock}}
    return
  }
}
