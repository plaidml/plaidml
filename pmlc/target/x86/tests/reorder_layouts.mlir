// RUN: pmlc-opt -x86-reorder-layouts -cse %s | FileCheck %s

#input  = affine_map<(n, h, w, c, r, s, k) -> (n, h + r, w + s, k)>
#filter = affine_map<(n, h, w, c, r, s, k) -> (r, s, k, c)>
#output = affine_map<(n, h, w, c, r, s, k) -> (n, h, w, c)>
#bias   = affine_map<(n, h, w, c) -> (c)>
#act    = affine_map<(n, h, w, c) -> (n, h, w, c)>

func @main(%arg0: tensor<1x56x56x64xf32>, %arg1: tensor<1x1x64x64xf32> {stdx.const}, %arg2: tensor<64xf32> {stdx.const}) -> tensor<1x56x56x64xf32> {
  %cst = arith.constant 0.000000e+00 : f32
  %T0 = linalg.init_tensor [1, 56, 56, 64] : tensor<1x56x56x64xf32>
  %1 = linalg.fill(%cst, %T0) : f32, tensor<1x56x56x64xf32> -> tensor<1x56x56x64xf32>

  // convolution
  %2 = linalg.generic {
    indexing_maps = [#input, #filter, #output],
    iterator_types = ["parallel", "parallel", "parallel", "parallel", "reduction", "reduction", "reduction"]
  } ins(%arg0, %arg1 : tensor<1x56x56x64xf32>, tensor<1x1x64x64xf32>) outs(%1 : tensor<1x56x56x64xf32>) {
  ^bb0(%arg3: f32, %arg4: f32, %arg5: f32):  // no predecessors
    %5 = arith.mulf %arg3, %arg4 : f32
    %6 = arith.addf %arg5, %5 : f32
    linalg.yield %6 : f32
  } -> tensor<1x56x56x64xf32>

  %T1 = linalg.init_tensor [1, 56, 56, 64] : tensor<1x56x56x64xf32>
  %3 = linalg.generic {
    indexing_maps = [#act, #bias, #act],
    iterator_types = ["parallel", "parallel", "parallel", "parallel"]
  } ins(%2, %arg2 : tensor<1x56x56x64xf32>, tensor<64xf32>) outs(%T1 : tensor<1x56x56x64xf32>) {
  ^bb0(%arg3: f32, %arg4: f32, %arg5: f32):  // no predecessors
    %5 = arith.addf %arg3, %arg4 : f32
    linalg.yield %5 : f32
  } -> tensor<1x56x56x64xf32>

  %T2 = linalg.init_tensor [1, 56, 56, 64] : tensor<1x56x56x64xf32>
  %4 = linalg.generic {
    indexing_maps = [#act, #act],
    iterator_types = ["parallel", "parallel", "parallel", "parallel"]
  } ins(%3 : tensor<1x56x56x64xf32>) outs(%T2 : tensor<1x56x56x64xf32>) {
  ^bb0(%arg3: f32, %arg4: f32):  // no predecessors
    %5 = stdx.relu(%arg3) : (f32) -> f32
    linalg.yield %5 : f32
  } -> tensor<1x56x56x64xf32>

  return %4 : tensor<1x56x56x64xf32>
}

// CHECK: #[[map0:.*]] = affine_map<(d0, d1, d2, d3, d4) -> (d0, d2, d3, d1 * 32 + d4)>
// CHECK: #[[map1:.*]] = affine_map<(d0, d1, d2, d3, d4) -> (d0, d1, d2, d3, d4)>
// CHECK: #[[map2:.*]] = affine_map<(d0, d1, d2, d3, d4, d5) -> (d2, d3, d0 * 32 + d4, d1 * 32 + d5)>
// CHECK: #[[map3:.*]] = affine_map<(d0, d1, d2, d3, d4, d5) -> (d0, d1, d2, d3, d4, d5)>
// CHECK: #[[map4:.*]] = affine_map<(d0, d1, d2, d3, d4, d5, d6, d7, d8) -> (d0, d8, d1 + d4, d2 + d5, d6)>
// CHECK: #[[map5:.*]] = affine_map<(d0, d1, d2, d3, d4, d5, d6, d7, d8) -> (d8, d7, d4, d5, d6, d3)>
// CHECK: #[[map6:.*]] = affine_map<(d0, d1, d2, d3, d4, d5, d6, d7, d8) -> (d0, d7, d1, d2, d3)>
// CHECK: #[[map7:.*]] = affine_map<(d0, d1, d2, d3, d4) -> (d1 * 32 + d4)>

//      CHECK: func @main(
// CHECK-SAME:   %[[arg0:.*]]: tensor<1x56x56x64xf32>
// CHECK-SAME:   %[[arg1:.*]]: tensor<1x1x64x64xf32> {stdx.const}
// CHECK-SAME:   %[[arg2:.*]]: tensor<64xf32> {stdx.const}
// CHECk-SAME:   -> tensor<1x56x56x64xf32>
//      CHECK:   linalg.init_tensor [1, 56, 56, 64] : tensor<1x56x56x64xf32>
//      CHECK:   linalg.fill
//      CHECK:   linalg.init_tensor [1, 2, 56, 56, 32] : tensor<1x2x56x56x32xf32>
//      CHECK:   linalgx.copy(%[[arg0]], %{{.*}}) {inputMap = #[[map0]], outputMap = #[[map1]]}
// CHECK-SAME:     : tensor<1x56x56x64xf32>, tensor<1x2x56x56x32xf32> -> tensor<1x2x56x56x32xf32>
//      CHECK:   linalg.init_tensor [2, 2, 1, 1, 32, 32] : tensor<2x2x1x1x32x32xf32>
//      CHECK:   linalgx.copy(%[[arg1]], %{{.*}}) {inputMap = #[[map2]], outputMap = #[[map3]]}
// CHECK-SAME:     : tensor<1x1x64x64xf32>, tensor<2x2x1x1x32x32xf32> -> tensor<2x2x1x1x32x32xf32>
//      CHECK:   linalgx.copy(%{{.*}}, %{{.*}}) {inputMap = #[[map0]], outputMap = #[[map1]]}
// CHECK-SAME:     : tensor<1x56x56x64xf32>, tensor<1x2x56x56x32xf32> -> tensor<1x2x56x56x32xf32>
//      CHECK:   linalg.generic
// CHECK-SAME:     indexing_maps = [#[[map4]], #[[map5]], #[[map6]]]
// CHECK-SAME:     iterator_types = ["parallel", "parallel", "parallel", "parallel", "reduction", "reduction", "reduction", "parallel", "reduction"]
// CHECK-SAME:     ins(%{{.*}}, %{{.*}} : tensor<1x2x56x56x32xf32>, tensor<2x2x1x1x32x32xf32>)
// CHECK-SAME:     outs(%{{.*}} : tensor<1x2x56x56x32xf32>)
//      CHECK:     mulf
//      CHECK:     addf
//      CHECK:     linalg.yield
//      CHECK:   } -> tensor<1x2x56x56x32xf32>
//      CHECK:   linalg.generic
// CHECK-SAME:     indexing_maps = [#[[map1]], #[[map7]], #[[map1]]]
// CHECK-SAME:     iterator_types = ["parallel", "parallel", "parallel", "parallel", "parallel"]
// CHECK-SAME:     ins(%{{.*}}, %[[arg2]] : tensor<1x2x56x56x32xf32>, tensor<64xf32>)
// CHECK-SAME:     outs(%{{.*}} : tensor<1x2x56x56x32xf32>)
//      CHECK:     addf
//      CHECK:     linalg.yield
//      CHECK:   } -> tensor<1x2x56x56x32xf32>
//      CHECK:   linalg.generic {
// CHECK-SAME:     indexing_maps = [#[[map1]], #[[map1]]]
// CHECK-SAME:     iterator_types = ["parallel", "parallel", "parallel", "parallel", "parallel"]
// CHECK-SAME:     ins(%{{.*}} : tensor<1x2x56x56x32xf32>) outs(%{{.*}} : tensor<1x2x56x56x32xf32>)
//      CHECK:     stdx.relu
//      CHECK:     linalg.yield
//      CHECK:   } -> tensor<1x2x56x56x32xf32>
//      CHECK:   linalgx.copy(%{{.*}}, %{{.*}}) {inputMap = #map1, outputMap = #map0}
// CHECK-SAME:     : tensor<1x2x56x56x32xf32>, tensor<1x56x56x64xf32> -> tensor<1x56x56x64xf32>
//      CHECK:   return %{{.*}} : tensor<1x56x56x64xf32>
