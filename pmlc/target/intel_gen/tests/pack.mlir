// RUN: pmlc-opt -intel-gen-affine-index-pack -split-input-file %s | FileCheck %s

// CHECK-DAG: #[[map0:.*]] = affine_map<(d0) -> (d0 mod 2)>
// CHECK-DAG: #[[map1:.*]] = affine_map<(d0) -> (d0 floordiv 8)>
// CHECK-DAG: #[[map2:.*]] = affine_map<(d0) -> ((d0 floordiv 2) mod 4)>
// CHECK-DAG: #[[map3:.*]] = affine_map<(d0) -> (d0)>
// CHECK-DAG: #[[map4:.*]] = affine_map<(d0) -> (d0 mod 5)>
// CHECK-DAG: #[[map5:.*]] = affine_map<(d0) -> (d0 floordiv 5)>
// CHECK-DAG: func @pack_test 
func @pack_test(%out: memref<1xf32>, %in: memref<100xf32>) {
  affine.parallel (%i0, %i1, %i2, %i3, %i4, %i5) = (0, 0, 0, 0, 0, 0) to (2, 3, 4, 7, 5,11) {
  // CHECK: affine.parallel (%[[d0:.*]], %[[d1:.*]], %[[d2:.*]]) = (0, 0, 0) to (24, 7, 55) {
    // CHECK-DAG: %[[i0:.*]] = affine.apply #[[map0]](%[[d0]])
    // CHECK-DAG: %[[i1:.*]] = affine.apply #[[map1]](%[[d0]])
    // CHECK-DAG: %[[i2:.*]] = affine.apply #[[map2]](%[[d0]])
    // CHECK-DAG: %[[i3:.*]] = affine.apply #[[map3]](%[[d1]])
    // CHECK-DAG: %[[i4:.*]] = affine.apply #[[map4]](%[[d2]])
    // CHECK-DAG: %[[i5:.*]] = affine.apply #[[map5]](%[[d2]])
    %a = affine.load %in[%i0 + %i1 + %i2 + %i3 + %i4 + %i5] :  memref<100xf32>
    // CHECK: affine.load %{{.*}}[%[[i0]] + %[[i1]] + %[[i2]] + %[[i3]] + %[[i4]] + %[[i5]]]
    affine.store %a, %out[0] : memref<1xf32>
  } {tags = {gpuBlock}}
  return
}

