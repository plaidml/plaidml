// RUN: pmlc-opt -tile-compute-bounds -convert-tile-to-pxa %s | FileCheck %s

#map0 = affine_map<(i, j, k) -> (j, k)>
#map1 = affine_map<(i, j, k) -> (j, i)>
#map2 = affine_map<(i, j, k) -> (i, k)>

func @dot(%arg0: tensor<1x784xf32>, %arg1: tensor<784x512xf32>) -> tensor<1x512xf32> {
  %c0 = "eltwise.sconst"() {value = 0.0 : f64} : () -> f32
  %0 = tile.constant 512
  %1 = tile.constant 1
  %2 = tile.contract add, mul, %c0, %arg0, %arg1 {sink=#map0, srcs=[#map1, #map2]} :
    f32, tensor<1x784xf32>, tensor<784x512xf32> -> tensor<1x512xf32>
  return %2 : tensor<1x512xf32>
}

// CHECK-DAG: [[map_lb:#map[0-9]+]] = affine_map<() -> (0, 0, 0)>
// CHECK-DAG: [[map_ub:#map[0-9]+]] = affine_map<() -> (784, 1, 512)>
// CHECK-LABEL: func @dot
// CHECK-SAME: %[[ARG0:.*]]: memref<1x784xf32>
// CHECK-SAME: %[[ARG1:.*]]: memref<784x512xf32>
// CHECK-SAME: -> memref<1x512xf32>
// CHECK-DAG: %[[ZERO:.*]] = constant 0.0 
// CHECK-DAG: %[[OUT:.*]] = alloc() : memref<1x512xf32>
// CHECK: %[[ZEROED:.*]] = affine.parallel  (%[[I1:.*]], %[[J1:.*]]) = (0, 0) to (1, 512)
// CHECK: pxa.reduce assign %[[ZERO]], %[[OUT]][%[[I1]], %[[J1]]] : memref<1x512xf32>
// CHECK: %[[FINAL:.*]] = affine.parallel (%[[I2:.*]], %[[J2:.*]], %[[K2:.*]]) = (0, 0, 0) to (784, 1, 512)
// CHECK-DAG: %[[A:.*]] = pxa.load %[[ARG0]][%[[J2]], %[[I2]]] : memref<1x784xf32>
// CHECK-DAG: %[[B:.*]] = pxa.load %[[ARG1]][%[[I2]], %[[K2]]] : memref<784x512xf32>
// CHECK:     %[[C:.*]] = mulf %[[A]], %[[B]] : f32
// CHECK:     pxa.reduce addf %[[C]], %[[ZEROED]][%[[J2]], %[[K2]]] : memref<1x512xf32>
// CHECK: return %[[FINAL]]
