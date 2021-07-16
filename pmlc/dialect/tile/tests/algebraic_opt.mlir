// RUN: pmlc-opt -tile-algebraic-opt -cse -split-input-file %s | FileCheck %s

#map2 = affine_map<(d0, d1, d2, d3, d4, d5, d6) -> (d0, d1, d2, d3)>
#map3 = affine_map<(d0, d1, d2, d3, d4, d5, d6) -> (d0, d1 * 2 + d4, d2 * 2 + d5, d6)>
#map4 = affine_map<(d0, d1, d2, d3, d4, d5, d6) -> (d4, d5, d6, d3)>

func @add_init(%I: tensor<1x230x230x3xf32>, %K: tensor<7x7x3x64xf32>, %B: tensor<64xf32>) -> tensor<1x112x112x64xf32> {
  %zero = tile.constant(0.0 : f64) : tensor<f32>
  %conv1 = tile.contract add, mul, %zero, %I, %K {sink = #map2, srcs = [#map3, #map4]} : tensor<f32>, tensor<1x230x230x3xf32>, tensor<7x7x3x64xf32> -> tensor<1x112x112x64xf32>
  %1 = tile.add %conv1, %B : (tensor<1x112x112x64xf32>, tensor<64xf32>) -> tensor<1x112x112x64xf32>
  %2 = tile.relu %1 : (tensor<1x112x112x64xf32>) -> tensor<1x112x112x64xf32>
  return %2 : tensor<1x112x112x64xf32>
}

// CHECK-LABEL: func @add_init
//  CHECK-SAME: (%[[I:.*]]: tensor<1x230x230x3xf32>, %[[K:.*]]: tensor<7x7x3x64xf32>, %[[B:.*]]: tensor<64xf32>)
//       CHECK: %[[O:.*]] = tile.contract add, mul, %[[B]], %[[I]], %[[K]]
//  CHECK-NEXT: tile.relu %[[O]]
