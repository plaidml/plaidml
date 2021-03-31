// RUN: pmlc-opt -convert-scf-to-omp -canonicalize -cse %s | FileCheck %s

module @get_value {
  // CHECK-LABEL: func @main
  func @main(%arg0: memref<1x1x64x64xf32>, %arg1: memref<1x56x56x64xf32>,
             %arg2: memref<1x56x56x64xf32>) {
    %cst = constant 0.000000e+00 : f32
    %c28 = constant 28 : index
    %c64 = constant 64 : index
    %c56 = constant 56 : index
    %c0 = constant 0 : index
    %c2 = constant 2 : index
    %c1 = constant 1 : index
    // CHECK: omp.parallel num_threads(%{{.*}} : index) default(shared)
    scf.parallel (%arg3) = (%c0) to (%c64) step (%c1) {
      scf.for %arg4 = %c0 to %c56 step %c1 {
        scf.for %arg5 = %c0 to %c56 step %c1 {
          store %cst, %arg2[%c0, %arg4, %arg5, %arg3] : memref<1x56x56x64xf32>
        }
      }
      // CHECK: omp.terminator
      scf.yield
    }
    %0 = xsmm.gemm.dispatch.f32 [28, 64, 64], [3584, 64, 3584]
    // CHECK: omp.parallel num_threads(%{{.*}} : index) default(shared)
    scf.parallel (%arg3) = (%c0) to (%c56) step (%c1) {
      scf.for %arg4 = %c0 to %c2 step %c1 {
        %1 = muli %arg4, %c28 : index
        xsmm.gemm.invoke.f32 %0, %arg2[%c0, %1, %arg3, %c0] = 
          %arg1[%c0, %1, %arg3, %c0], %arg0[%c0, %c0, %c0, %c0] : 
          (memref<1x56x56x64xf32>, memref<1x1x64x64xf32>) -> 
          memref<1x56x56x64xf32>
      }
      // CHECK: omp.terminator
      scf.yield
    }
    return
  }
}
