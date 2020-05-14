// RUN: pmlc-opt --target-intel_gen %s | FileCheck %s

#map0 = affine_map<() -> (0, 0, 0)>
#map1 = affine_map<(d0, d1, d2) -> (d0, d1)>
#map2 = affine_map<(d0, d1, d2) -> (d0, d2)>
#map3 = affine_map<(d0, d1, d2) -> (d2, d1)>
#map4 = affine_map<() -> (2, 2, 2)>


module {
  func @dot(%arg0: tensor<3x3xf32>, %arg1: tensor<3x3xf32>) -> tensor<3x3xf32> {
    %cst = "eltwise.sconst"() {value = 0.000000e+00 : f64} : () -> tensor<f32>
    %0 = tile.contract add, mul, %cst, %arg1, %arg0 {idxs = ["i", "j", "k"], lowerBounds = #map0, sink = #map1, srcs = [#map2, #map3], upperBounds = #map4} : tensor<f32>, tensor<3x3xf32>, tensor<3x3xf32> -> tensor<3x3xf32>
    return %0 : tensor<3x3xf32>
  }
}

// CHECK: llvm.call @submitCommandBuffers
