// RUN: pmlc-opt --linalgx-regulate-depthwise %s | FileCheck %s

#map0 = affine_map<(d0, d1, d2, d3, d4, d5, d6, d7) -> (d0, d1 + d6, d2 + d7, d3 + d4 - d5)>
#map1 = affine_map<(d0, d1, d2, d3, d4, d5, d6, d7) -> (d6, d7, d3 + d4 - d5, d5)>
#map2 = affine_map<(d0, d1, d2, d3, d4, d5, d6, d7) -> (d0, d1, d2, d3)>

// CHECK: #map0 = affine_map<(d0, d1, d2, d3, d4, d5, d6) -> (d0, d1 + d4, d2 + d5, d3)>
// CHECK: #map1 = affine_map<(d0, d1, d2, d3, d4, d5, d6) -> (d4, d5, d3, d6)>
// CHECK: #map2 = affine_map<(d0, d1, d2, d3, d4, d5, d6) -> (d0, d1, d2, d3)>

func @main(%arg0: tensor<1x114x114x32xf32>, %arg1: tensor<3x3x32x1xf32>, %arg2: tensor<1x112x112x32xf32>) -> tensor<1x112x112x32xf32>{
    %0 = linalg.generic {indexing_maps = [#map0, #map1, #map2], iterator_types = ["parallel", "parallel", "parallel", "parallel", "reduction", "reduction", "reduction", "reduction"]} ins(%arg0, %arg1 : tensor<1x114x114x32xf32>, tensor<3x3x32x1xf32>) outs(%arg2 : tensor<1x112x112x32xf32>) attrs =  {iterator_ranges = [1, 112, 112, 32, 1, 1, 3, 3]} {
// CHECK: linalg.generic
// CHECK-SAME: iterator_types = ["parallel", "parallel", "parallel", "parallel", "reduction", "reduction", "reduction"]
// CHECK-SAME: iterator_ranges = [1, 112, 112, 32, 3, 3, 1]
    ^bb0(%arg3: f32, %arg4: f32, %arg5: f32):  // no predecessors
      %1 = arith.mulf %arg3, %arg4 : f32
      %2 = arith.addf %arg5, %1 : f32
      linalg.yield %2 : f32
    } -> tensor<1x112x112x32xf32>
    return %0 : tensor<1x112x112x32xf32>
}
