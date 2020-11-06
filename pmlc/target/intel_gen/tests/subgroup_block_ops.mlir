// RUN: pmlc-opt -stdx-subgroup-broadcast=use-block-ops=true -cse %s | FileCheck %s 

// CHECK-LABEL: @subgroup_block_read_write
func @subgroup_block_read_write(%arg0: memref<64xf32>, %arg1: memref<64xf32>) {
  %c0 = constant 0 : index
  %c8 = constant 8 : index
  %c64 = constant 64 : index
  %cst = constant 0.000000e+00 : f32
  %c1 = constant 1 : index
  // CHECK: scf.parallel (%{{.*}}) = (%{{.*}})
  scf.parallel (%j) = (%c0) to (%c8) step (%c1) {
    // CHECK:   scf.parallel (%{{.*}}, %[[IDX:.*]]) = (%{{.*}}, %{{.*}})
    scf.parallel (%i) = (%c0) to (%c64) step (%c8) {
      // CHECK: stdx.subgroup_block_read_intel %{{.*}}[%[[IDX]]] : memref<64xf32>
      %0 = vector.transfer_read %arg0[%i], %cst : memref<64xf32>, vector<8xf32>
      %1 = extract_element %0[%c1] : vector<8xf32>
      %2 = vector.broadcast %1 : f32 to vector<8xf32>
      // CHECK: stdx.subgroup_block_write_intel %{{.*}}, %{{.*}}[%[[IDX]]] : f32, memref<64xf32>
      vector.transfer_write %2, %arg1[%i] : vector<8xf32>, memref<64xf32>
      scf.yield
    }  {tags = {gpuThread, subgroupSize=8}}
  }  {tags = {gpuBlock, subgroupSize=8}}
  return
}
