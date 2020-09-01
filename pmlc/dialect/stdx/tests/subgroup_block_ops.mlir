// RUN: pmlc-opt -stdx-subgroup-block-ops %s | FileCheck %s

// CHECK-LABEL: @subgroup_block_read_write
func @subgroup_block_read_write(%arg0: memref<64xf32>, %arg1: memref<64xf32>) {
  %c0 = constant 0 : index
  %c8 = constant 8 : index
  %c64 = constant 64 : index
  %cst = constant 0.000000e+00 : f32
  %c1 = constant 1 : index
  // CHECK:   scf.parallel (%[[IDX:.*]], %{{.*}}) = (%{{.*}}, %{{.*}})
  scf.parallel (%arg2, %arg3) = (%c0, %c0) to (%c64, %c8) step (%c8, %c1) {
    %0 = addi %arg2, %arg3 : index
    // CHECK: %[[READ:.*]] = stdx.subgroup_block_read_intel %{{.*}}[%[[IDX]]] : memref<64xf32>
    %1 = load %arg0[%0] : memref<64xf32>
    // CHECK: stdx.subgroup_block_write_intel %[[READ]], %{{.*}}[%[[IDX]]] : memref<64xf32>
    store %1, %arg1[%0] : memref<64xf32>
    scf.yield
  }
  return
}
