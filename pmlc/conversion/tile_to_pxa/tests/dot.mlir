// RUN: pmlc-opt -tile-compute-bounds -convert-tile-to-pxa -canonicalize -cse %s | FileCheck %s

#map0 = affine_map<(i, j, k) -> (j, k)>
#map1 = affine_map<(i, j, k) -> (j, i)>
#map2 = affine_map<(i, j, k) -> (i, k)>

!f32 = type f32
!i32 = type !eltwise.i32
func @dot(%arg0: tensor<1x784xf32>, %arg1: tensor<784x512xf32>) -> tensor<1x512xf32> {
  %c0 = "eltwise.sconst"() {value = 0.0 : f64} : () -> !f32
  %0 = tile.constant 512
  %1 = tile.constant 1
  %2 = tile.contract add, mul, %c0, %arg0, %arg1 {sink=#map0, srcs=[#map1, #map2]} :
    !f32, tensor<1x784xf32>, tensor<784x512xf32> -> tensor<1x512xf32>
  return %2 : tensor<1x512xf32>
}

// CHECK-DAG: [[map_lb:#map[0-9]+]] = affine_map<() -> (0, 0, 0)>
// CHECK-DAG: [[map_ub:#map[0-9]+]] = affine_map<() -> (784, 1, 512)>
// CHECK-LABEL: func @dot
// CHECK-SAME: %[[ARG0:.*]]: memref<1x784xf32>
// CHECK-SAME: %[[ARG1:.*]]: memref<784x512xf32>
// CHECK-SAME: %[[ARG2:.*]]: memref<1x512xf32>
// CHECK: affine.parallel (%[[I:.*]], %[[J:.*]], %[[K:.*]]) = (0, 0, 0) to (784, 1, 512)
// CHECK-DAG: %[[A:.*]] = affine.load %[[ARG0]][%[[J]], %[[I]]] : memref<1x784xf32>
// CHECK-DAG: %[[B:.*]] = affine.load %[[ARG1]][%[[I]], %[[K]]] : memref<784x512xf32>
// CHECK:     %[[C:.*]] = mulf %[[A]], %[[B]] : f32
// CHECK:     pxa.reduce add %[[C]], %[[ARG2]][%[[J]], %[[K]]] : memref<1x512xf32>
