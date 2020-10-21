// RUN: pmlc-opt -tile-compute-bounds -tile-pad-ranges %s | FileCheck %s

#map0 = affine_map<(d0, d1, d2, d3, d4, d5, d6) -> (d0, d1, d2, d3)>
#map1 = affine_map<(d0, d1, d2, d3, d4, d5, d6) -> (d0, d1 * 2 + d4, d2 * 2 + d5, d6)>
#map2 = affine_map<(d0, d1, d2, d3, d4, d5, d6) -> (d4, d5, d6, d3)>

func @resnetLayer1(%arg0: tensor<7x7x3x64xf32>, %arg1: tensor<1x224x224x3xf32>) -> tensor<1x109x109x64xf32> {
  %cst = tile.constant(0.000000e+00 : f64) : tensor<f32>
  %conv = tile.contract add, mul, %cst, %arg1, %arg0 {sink = #map0, srcs = [#map1, #map2]}
    : tensor<f32>, tensor<1x224x224x3xf32>, tensor<7x7x3x64xf32> -> tensor<1x109x109x64xf32>
  return %conv : tensor<1x109x109x64xf32>
}

// CHECK: affine_map<() -> (0, 111, 111, 63, 6, 6, 2)>
