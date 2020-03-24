// RUN: pmlc-opt -tile-compute-bounds -cse -split-input-file %s | FileCheck %s

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

// CHECK: #map0 = affine_map<() -> (0, 0, 0)>
// CHECK: #map1 = affine_map<(d0, d1, d2) -> (d1, d2)>
// CHECK: #map2 = affine_map<(d0, d1, d2) -> (d1, d0)>
// CHECK: #map3 = affine_map<(d0, d1, d2) -> (d0, d2)>
// CHECK: #map4 = affine_map<() -> (783, 0, 511)>
// CHECK-LABEL: func @dot
// CHECK: tile.contract
// CHECK-SAME: lowerBounds = #map0
// CHECK-SAME: sink = #map1
// CHECK-SAME: srcs = [#map2, #map3]
// CHECK-SAME: upperBounds = #map4

// -----
