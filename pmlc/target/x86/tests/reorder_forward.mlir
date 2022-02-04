// RUN: pmlc-opt -x86-reorder-layouts -cse %s | FileCheck %s

#map1 = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>
#map2 = affine_map<(d0, d1, d2, d3, d4) -> (d0, d2, d3, d1 * 32 + d4)>
#map3 = affine_map<(d0, d1, d2, d3, d4) -> (d0, d1, d2, d3, d4)>
#map4 = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3 + 64)>

func @inception_v3(%arg0: tensor<1x2x35x35x32xf32>) -> tensor<1x35x35x256xf32> {
  %cst = constant 0.000000e+00 : f32
  %0 = linalg.init_tensor [1, 35, 35, 64] : tensor<1x35x35x64xf32>
  %1 = linalgx.copy(%arg0, %0) {inputMap = #map3, outputMap = #map2}
    : tensor<1x2x35x35x32xf32>, tensor<1x35x35x64xf32> -> tensor<1x35x35x64xf32>
  %2 = linalg.init_tensor [1, 35, 35, 256] : tensor<1x35x35x256xf32>
  %3 = linalg.fill(%cst, %2) : f32, tensor<1x35x35x256xf32> -> tensor<1x35x35x256xf32>
  %4 = linalg.generic
    {indexing_maps = [#map1, #map1], iterator_types = ["parallel", "parallel", "parallel", "parallel"]}
    ins(%1 : tensor<1x35x35x64xf32>) outs(%3 : tensor<1x35x35x256xf32>) attrs = {iterator_ranges = [1, 35, 35, 64]} {
  ^bb0(%arg5: f32, %arg6: f32):  // no predecessors
    linalg.yield %arg5 : f32
  } -> tensor<1x35x35x256xf32>
  return %4 : tensor<1x35x35x256xf32>
}

// CHECK: func @inception_v3
// CHECK:   linalg.init_tensor
// CHECK:   linalgx.copy
// CHECK:   linalg.init_tensor
// CHECK:   linalg.fill
// CHECK:   linalg.generic
// CHECK:   return
