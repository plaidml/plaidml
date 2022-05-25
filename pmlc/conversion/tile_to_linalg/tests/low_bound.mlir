// RUN: pmlc-opt -convert-tile-to-linalg %s | FileCheck %s

#map0 = affine_map<() -> (0, 0, 1, 0, 0, 0, 0)>
#map1 = affine_map<(d0, d1, d2, d3, d4, d5, d6) -> (d0, d1, d2 - 1, d3)>
#map2 = affine_map<(d0, d1, d2, d3, d4, d5, d6) -> (d0, d4, d6, d5)>
#map3 = affine_map<(d0, d1, d2, d3, d4, d5, d6) -> (d1 - d4, d2 - d6 * 4, d3, d5)>
#map4 = affine_map<() -> (0, 0, 9, 0, 0, 0, 2)>
#set = affine_set<(d0, d1, d2, d3, d4, d5, d6) : (d2 - d6 * 4 >= 0, -d2 + d6 * 4 + 3 >= 0)>

func.func @main(%arg0: tensor<1x1x3x1xf32> {stdx.const}, %arg1: tensor<1x4x1x1xf32> {stdx.const}) -> tensor<1x1x9x1xf32> {
  %cst = tile.constant(0.000000e+00 : f64) : tensor<f32>
  %conv = tile.contract add, mul, %cst, %arg0, %arg1 {
    cons = #set, lowerBounds = #map0, sink = #map1, srcs = [#map2, #map3], upperBounds = #map4
  } : tensor<f32>, tensor<1x1x3x1xf32>, tensor<1x4x1x1xf32> -> tensor<1x1x9x1xf32>
  return %conv : tensor<1x1x9x1xf32>
}

// CHECK: #[[map0:.*]] = affine_map<(d0, d1, d2, d3, d4, d5, d6) -> (d0, d4, d6, d5)>
// CHECK: #[[map1:.*]] = affine_map<(d0, d1, d2, d3, d4, d5, d6) -> (d1 - d4, d2 - d6 * 4 + 1, d3, d5)>
// CHECK: #[[map2:.*]] = affine_map<(d0, d1, d2, d3, d4, d5, d6) -> (d0, d1, d2, d3)>
// CHECK: #[[set:.*]] = affine_set<(d0, d1, d2, d3, d4, d5, d6) : (d2 - d6 * 4 + 1 >= 0, -(d2 + 1) + d6 * 4 + 3 >= 0)>
// CHECK: linalg.generic {indexing_maps = [#[[map0]], #[[map1]], #[[map2]]]
// CHECK-SAME: constraints = #[[set]]
