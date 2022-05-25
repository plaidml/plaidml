// RUN: pmlc-opt -split-input-file -x86-reorder-layouts -cse %s | FileCheck %s

#map1 = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>
#map2 = affine_map<(d0, d1, d2, d3, d4) -> (d0, d2, d3, d1 * 32 + d4)>
#map3 = affine_map<(d0, d1, d2, d3, d4) -> (d0, d1, d2, d3, d4)>

func.func @inception_v3(%arg0: tensor<1x2x35x35x32xf32>) -> tensor<1x35x35x256xf32> {
  %cst = arith.constant 0.000000e+00 : f32
  %0 = linalg.init_tensor [1, 35, 35, 64] : tensor<1x35x35x64xf32>
  %1 = linalgx.copy(%arg0, %0) {inputMap = #map3, outputMap = #map2}
    : tensor<1x2x35x35x32xf32>, tensor<1x35x35x64xf32> -> tensor<1x35x35x64xf32>
  %2 = linalg.init_tensor [1, 35, 35, 256] : tensor<1x35x35x256xf32>
  %3 = linalg.fill ins(%cst : f32) outs(%2 : tensor<1x35x35x256xf32>) -> tensor<1x35x35x256xf32>
  %4 = linalg.generic
    {indexing_maps = [#map1, #map1], iterator_types = ["parallel", "parallel", "parallel", "parallel"]}
    ins(%1 : tensor<1x35x35x64xf32>) outs(%3 : tensor<1x35x35x256xf32>) attrs = {iterator_ranges = [1, 35, 35, 64]} {
  ^bb0(%arg5: f32, %arg6: f32):  // no predecessors
    linalg.yield %arg5 : f32
  } -> tensor<1x35x35x256xf32>
  return %4 : tensor<1x35x35x256xf32>
}

// CHECK: func.func @inception_v3
// CHECK:   linalg.init_tensor
// CHECK:   linalgx.copy
// CHECK:   linalg.init_tensor
// CHECK:   linalg.fill
// CHECK:   linalg.generic
// CHECK:   return

// -----

#map0 = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>
#map1 = affine_map<(d0, d1, d2, d3) -> (d3)>
#map2 = affine_map<(d0, d1, d2, d3, d4, d5, d6) -> (d4, d5, d6, d3)>
#map3 = affine_map<(d0, d1, d2, d3, d4, d5, d6) -> (d0, d1, d2, d3)>
#map4 = affine_map<(d0, d1, d2, d3, d4, d5, d6) -> (d0, d1 + d4, d2 + d5, d6)>

func.func @broadcast(%arg0: tensor<64xf32>, %arg1: tensor<1x56x56x64xf32>, %arg2: tensor<1x1x64x64xf32>) -> tensor<1x56x56x64xf32> {
  %0 = linalg.init_tensor [1, 56, 56, 64] : tensor<1x56x56x64xf32>
  // broadcast
  %1 = linalg.generic
    {indexing_maps = [#map1, #map0], iterator_types = ["parallel", "parallel", "parallel", "parallel"]}
    ins(%arg0 : tensor<64xf32>) outs(%0 : tensor<1x56x56x64xf32>) {
  ^bb0(%arg3: f32, %arg4: f32):  // no predecessors
    linalg.yield %arg3 : f32
  } -> tensor<1x56x56x64xf32>
  // convolution
  %2 = linalg.generic {
    indexing_maps = [#map4, #map2, #map3],
    iterator_types = ["parallel", "parallel", "parallel", "parallel", "reduction", "reduction", "reduction"]
  } ins(%arg1, %arg2 : tensor<1x56x56x64xf32>, tensor<1x1x64x64xf32>)
    outs(%1 : tensor<1x56x56x64xf32>)
    attrs = {iterator_ranges = [1, 56, 56, 64, 1, 1, 64]} {
  ^bb0(%arg3: f32, %arg4: f32, %arg5: f32):  // no predecessors
    %334 = arith.mulf %arg3, %arg4 : f32
    %335 = arith.addf %arg5, %334 : f32
    linalg.yield %335 : f32
  } -> tensor<1x56x56x64xf32>
  return %2 : tensor<1x56x56x64xf32>
}

//      CHECK: #[[map0:.*]] = affine_map<(d0, d1, d2, d3, d4) -> (d0, d1, d2, d3, d4)>
//      CHECK: #[[map1:.*]] = affine_map<(d0, d1, d2, d3, d4) -> (d1 * 32 + d4)>
//      CHECK: func.func @broadcast
//      CHECK:   %[[BROADCAST:.*]] = linalg.generic
// CHECK-SAME:     indexing_maps = [#[[map1]], #[[map0]]]
// CHECK-SAME:     iterator_types = ["parallel", "parallel", "parallel", "parallel", "parallel"]
// CHECK-SAME:     ins(%{{.*}} : tensor<64xf32>)
// CHECK-SAME:     outs(%{{.*}} : tensor<1x2x56x56x32xf32>)
//      CHECK:     linalg.yield
//      CHECK:   } -> tensor<1x2x56x56x32xf32>
//      CHECK:   linalg.generic
// CHECK-SAME:     outs(%[[BROADCAST]] : tensor<1x2x56x56x32xf32>)
//      CHECK:     mulf
//      CHECK:     addf
//      CHECK:     linalg.yield
