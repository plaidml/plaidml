// RUN: pmlc-opt -convert-tile-to-linalg %s | FileCheck %s

#map0 = affine_map<() -> (0, 0, 0, 0, 0, 0, 0, 0, 0)>
#map1 = affine_map<(d0, d1, d2, d3, d4, d5, d6, d7, d8) -> (d0, d1, d2 + d3 * 2 - 1, d4 + d5 * 2 - 1)>
#map2 = affine_map<(d0, d1, d2, d3, d4, d5, d6, d7, d8) -> (d0, d8, d3 - d6, d5 - d7)>
#map3 = affine_map<(d0, d1, d2, d3, d4, d5, d6, d7, d8) -> (d2 + d6 * 2, d4 + d7 * 2, d1, d8)>
#map4 = affine_map<() -> (3, 1, 1, 4, 1, 4, 1, 1, 7)>
#set = affine_set<(d0, d1, d2, d3, d4, d5, d6, d7, d8) : (d2 + d3 * 2 - 1 >= 0, d4 + d5 * 2 - 1 >= 0)>

func.func @neg_bound(%arg0: tensor<4x8x5x5xf32> {stdx.const}, %arg1: tensor<3x3x2x8xf32> {stdx.const}) -> tensor<4x2x9x9xf32> {
  %cst = tile.constant(0.000000e+00 : f64) : tensor<f32>
  %0 = tile.ident %arg1 {padLower = [0, 0, 0, 0], padType = 1 : i64, padUpper = [1, 1, 0, 0]} : (tensor<3x3x2x8xf32>) -> tensor<3x3x2x8xf32>
  %1 = tile.ident %arg0 {padLower = [0, 0, 1, 1], padType = 1 : i64, padUpper = [0, 0, 0, 0]} : (tensor<4x8x5x5xf32>) -> tensor<4x8x5x5xf32>
  %conv = tile.contract add, mul, %cst, %1, %0 {cons = #set, lowerBounds = #map0, sink = #map1, srcs = [#map2, #map3], upperBounds = #map4} : tensor<f32>, tensor<4x8x5x5xf32>, tensor<3x3x2x8xf32> -> tensor<4x2x9x9xf32>
  return %conv : tensor<4x2x9x9xf32>
}

// CHECK-LABEL: func.func @neg_bound
//       CHECK:   linalg.generic
//       CHECK:   linalg.generic
//       CHECK:   linalg.generic
