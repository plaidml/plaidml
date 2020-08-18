// RUN: pmlc-opt -stdx-subgroup-broadcast %s | FileCheck %s

func @subgroup_test(%arg0: memref<64xf32>, %arg1: memref<64xf32>) {
  %c1_i32 = constant 1 : i32
  %c0 = constant 0 : index
  %c8 = constant 8 : index
  %c64 = constant 64 : index
  %cst = constant 0.000000e+00 : f32
  // CHECK: constant 1 : index
  // CHECK: scf.parallel (%[[I:.*]], %[[SID:.*]]) = (%{{.*}}, %{{.*}}) to (%{{.*}}, %{{.*}}) step (%{{.*}}, %{{.*}}) {
  // CHECK-NEXT:   %[[IDX:.*]] = addi %[[I]], %[[SID]] : index
  // CHECK-NEXT:   %[[LOAD:.*]] = load %{{.*}}[%[[IDX]]] : memref<64xf32>
  // CHECK-NEXT:   %[[BROADCAST:.*]] = stdx.subgroup_broadcast(%[[LOAD]], %{{.*}}) : f32, i32
  // CHECK-NEXT:   store %[[BROADCAST]], %{{.*}}[%[[IDX]]] : memref<64xf32>
  scf.parallel (%i) = (%c0) to (%c64) step (%c8) {
    %0 = vector.transfer_read %arg0[%i], %cst : memref<64xf32>, vector<8xf32>
    %1 = vector.extractelement %0[%c1_i32 : i32] : vector<8xf32>
    %2 = vector.broadcast %1 : f32 to vector<8xf32>
    vector.transfer_write %2, %arg1[%i] : vector<8xf32>, memref<64xf32>
    scf.yield
  }
  return
}
