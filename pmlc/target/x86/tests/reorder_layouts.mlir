// RUN: pmlc-opt -split-input-file -x86-reorder-layouts -cse %s | FileCheck %s

#input  = affine_map<(n, h, w, c, r, s, k) -> (n, h + r, w + s, k)>
#filter = affine_map<(n, h, w, c, r, s, k) -> (r, s, k, c)>
#output = affine_map<(n, h, w, c, r, s, k) -> (n, h, w, c)>
#bias   = affine_map<(n, h, w, c) -> (c)>
#act    = affine_map<(n, h, w, c) -> (n, h, w, c)>

func @main(%arg0: tensor<1x56x56x64xf32>, %arg1: tensor<1x1x64x64xf32> {stdx.const}, %arg2: tensor<64xf32> {stdx.const}) -> tensor<1x56x56x64xf32> {
  %cst = constant 0.000000e+00 : f32
  %0 = linalg.init_tensor [1, 56, 56, 64] : tensor<1x56x56x64xf32>
  %1 = linalg.fill(%cst, %0) : f32, tensor<1x56x56x64xf32> -> tensor<1x56x56x64xf32>

  %2 = linalg.generic {
    indexing_maps = [#input, #filter, #output],
    iterator_types = ["parallel", "parallel", "parallel", "parallel", "reduction", "reduction", "reduction"]
  } ins(%arg0, %arg1 : tensor<1x56x56x64xf32>, tensor<1x1x64x64xf32>) outs(%1 : tensor<1x56x56x64xf32>) {
  ^bb0(%arg3: f32, %arg4: f32, %arg5: f32):  // no predecessors
    %5 = mulf %arg3, %arg4 : f32
    %6 = addf %arg5, %5 : f32
    linalg.yield %6 : f32
  } -> tensor<1x56x56x64xf32>

  %3 = linalg.generic {
    indexing_maps = [#act, #bias, #act],
    iterator_types = ["parallel", "parallel", "parallel", "parallel"]
  } ins(%2, %arg2 : tensor<1x56x56x64xf32>, tensor<64xf32>) outs(%0 : tensor<1x56x56x64xf32>) {
  ^bb0(%arg3: f32, %arg4: f32, %arg5: f32):  // no predecessors
    %5 = addf %arg3, %arg4 : f32
    linalg.yield %5 : f32
  } -> tensor<1x56x56x64xf32>

  %4 = linalg.generic {
    indexing_maps = [#act, #act],
    iterator_types = ["parallel", "parallel", "parallel", "parallel"]
  } ins(%3 : tensor<1x56x56x64xf32>) outs(%0 : tensor<1x56x56x64xf32>) {
  ^bb0(%arg3: f32, %arg4: f32):  // no predecessors
    %5 = stdx.relu(%arg3) : (f32) -> f32
    linalg.yield %5 : f32
  } -> tensor<1x56x56x64xf32>

  return %4 : tensor<1x56x56x64xf32>
}

// CHECK: #[[map0:.*]] = affine_map<(d0, d1, d2, d3, d4) -> (d0, d2, d3, d1 * 16 + d4)>
// CHECK: #[[map1:.*]] = affine_map<(d0, d1, d2, d3, d4) -> (d0, d1, d2, d3, d4)>
// CHECK: #[[map2:.*]] = affine_map<(d0, d1, d2, d3, d4, d5) -> (d2, d3, d0 * 16 + d4, d1 * 16 + d5)>
// CHECK: #[[map3:.*]] = affine_map<(d0, d1, d2, d3, d4, d5) -> (d0, d1, d2, d3, d4, d5)>
// CHECK: #[[map4:.*]] = affine_map<(d0, d1, d2, d3, d4, d5, d6, d7, d8) -> (d0, d8, d1 + d4, d2 + d5, d6)>
// CHECK: #[[map5:.*]] = affine_map<(d0, d1, d2, d3, d4, d5, d6, d7, d8) -> (d8, d7, d4, d5, d6, d3)>
// CHECK: #[[map6:.*]] = affine_map<(d0, d1, d2, d3, d4, d5, d6, d7, d8) -> (d0, d7, d1, d2, d3)>
// CHECK: #[[map7:.*]] = affine_map<(d0, d1, d2, d3, d4) -> (d1 * 16 + d4)>

//       CHECK: func @main(
//  CHECK-SAME:   %[[arg0:.*]]: tensor<1x56x56x64xf32>
//  CHECK-SAME:   %[[arg1:.*]]: tensor<1x1x64x64xf32> {stdx.const}
//  CHECK-SAME:   %[[arg2:.*]]: tensor<64xf32> {stdx.const}) -> tensor<1x56x56x64xf32>
//       CHECK:   %[[zero:.*]] = constant 0.000000e+00 : f32
// reorder input
//       CHECK:   %[[X0:.*]] = linalg.init_tensor [1, 4, 56, 56, 16] : tensor<1x4x56x56x16xf32>
//       CHECK:   %[[X1:.*]] = linalg.generic
//  CHECK-SAME:     indexing_maps = [#[[map0]], #[[map1]]]
//  CHECK-SAME:     iterator_types = ["parallel", "parallel", "parallel", "parallel", "parallel"]
//  CHECK-SAME:     ins(%[[arg0]] : tensor<1x56x56x64xf32>)
//  CHECK-SAME:     outs(%[[X0]] : tensor<1x4x56x56x16xf32>)
//       CHECK:   ^bb0(%[[arg3:[a-z0-9]+]]: f32, %[[arg4:[a-z0-9]+]]: f32):  // no predecessors
//       CHECK:     linalg.yield %[[arg3]] : f32
//       CHECK:   } -> tensor<1x4x56x56x16xf32>
// reorder filter
//       CHECK:   %[[X2:.*]] = linalg.init_tensor [4, 4, 1, 1, 16, 16] : tensor<4x4x1x1x16x16xf32>
//       CHECK:   %[[X3:.*]] = linalg.generic
//  CHECK-SAME:     indexing_maps = [#[[map2]], #[[map3]]]
//  CHECK-SAME:     iterator_types = ["parallel", "parallel", "parallel", "parallel", "parallel", "parallel"]
//  CHECK-SAME:     ins(%[[arg1]] : tensor<1x1x64x64xf32>)
//  CHECK-SAME:     outs(%[[X2]] : tensor<4x4x1x1x16x16xf32>)
//       CHECK:   ^bb0(%[[arg3:[a-z0-9]+]]: f32, %[[arg4:[a-z0-9]+]]: f32):  // no predecessors
//       CHECK:     linalg.yield %[[arg3]] : f32
//       CHECK:   } -> tensor<4x4x1x1x16x16xf32>
// convolution
//       CHECK:   %[[X4:.*]] = linalg.fill(%[[zero]], %[[X0]]) : f32, tensor<1x4x56x56x16xf32> -> tensor<1x4x56x56x16xf32>
//       CHECK:   %[[X5:.*]] = linalg.generic
//  CHECK-SAME:     indexing_maps = [#[[map4]], #[[map5]], #[[map6]]]
//  CHECK-SAME:     iterator_types = ["parallel", "parallel", "parallel", "parallel", "reduction", "reduction", "reduction", "parallel", "reduction"]
//  CHECK-SAME:     ins(%[[X1]], %[[X3]] : tensor<1x4x56x56x16xf32>, tensor<4x4x1x1x16x16xf32>)
//  CHECK-SAME:     outs(%[[X4]] : tensor<1x4x56x56x16xf32>)
//       CHECK:   ^bb0(%[[arg3:[a-z0-9]+]]: f32, %[[arg4:[a-z0-9]+]]: f32, %[[arg5:[a-z0-9]+]]: f32):  // no predecessors
//       CHECK:     %[[X10:.*]] = mulf %[[arg3]], %[[arg4]] : f32
//       CHECK:     %[[X11:.*]] = addf %[[arg5]], %[[X10]] : f32
//       CHECK:     linalg.yield %[[X11]] : f32
//       CHECK:   } -> tensor<1x4x56x56x16xf32>
// bias add
//       CHECK:   %[[X6:.*]] = linalg.init_tensor [1, 56, 56, 64] : tensor<1x56x56x64xf32>
//       CHECK:   %[[X7:.*]] = linalg.generic
//  CHECK-SAME:     indexing_maps = [#[[map1]], #[[map7]], #[[map1]]]
//  CHECK-SAME:     iterator_types = ["parallel", "parallel", "parallel", "parallel", "parallel"]
//  CHECK-SAME:     ins(%[[X5]], %[[arg2]] : tensor<1x4x56x56x16xf32>, tensor<64xf32>)
//  CHECK-SAME:     outs(%[[X0]] : tensor<1x4x56x56x16xf32>)
//       CHECK:     addf
//       CHECK:     linalg.yield
//       CHECK:   } -> tensor<1x4x56x56x16xf32>
// relu
//       CHECK:   %[[X8:.*]] = linalg.generic
//  CHECK-SAME:     indexing_maps = [#[[map1]], #[[map1]]]
//  CHECK-SAME:     iterator_types = ["parallel", "parallel", "parallel", "parallel", "parallel"]
//  CHECK-SAME:     ins(%[[X7]] : tensor<1x4x56x56x16xf32>)
//  CHECK-SAME:     outs(%[[X0]] : tensor<1x4x56x56x16xf32>)
//       CHECK:     stdx.relu
//       CHECK:     linalg.yield
//       CHECK:   } -> tensor<1x4x56x56x16xf32>
// reorder output
//       CHECK:   %[[X9:.*]] = linalg.generic
//  CHECK-SAME:     indexing_maps = [#[[map1]], #[[map0]]]
//  CHECK-SAME:     iterator_types = ["parallel", "parallel", "parallel", "parallel", "parallel"]
//  CHECK-SAME:     ins(%[[X8]] : tensor<1x4x56x56x16xf32>)
//  CHECK-SAME:     outs(%[[X6]] : tensor<1x56x56x64xf32>)
//       CHECK:   ^bb0(%[[arg3:[a-z0-9]+]]: f32, %[[arg4:[a-z0-9]+]]: f32):  // no predecessors
//       CHECK:     linalg.yield %[[arg3]] : f32
//       CHECK:   } -> tensor<1x56x56x64xf32>
//       CHECK:   return %[[X9]] : tensor<1x56x56x64xf32>

// -----

#map0 = affine_map<(d0, d1, d2, d3, d4) -> (d0, d2, d3, d1 * 16 + d4)>
#map1 = affine_map<(d0, d1, d2, d3, d4) -> (d0, d1, d2, d3, d4)>

func @reorder_fold(%arg0: tensor<1x4x56x56x16xf32>) -> tensor<1x4x56x56x16xf32> {
  %0 = linalg.init_tensor [1, 56, 56, 64] : tensor<1x56x56x64xf32>
  %1 = linalg.generic {
    indexing_maps = [#map1, #map0],
    iterator_types = ["parallel", "parallel", "parallel", "parallel", "parallel"]
  } ins(%arg0 : tensor<1x4x56x56x16xf32>) outs(%0 : tensor<1x56x56x64xf32>) attrs = {reorder} {
  ^bb0(%arg2: f32, %arg3: f32):  // no predecessors
    linalg.yield %arg2 : f32
  } -> tensor<1x56x56x64xf32>

  %2 = linalg.init_tensor [1, 4, 56, 56, 16] : tensor<1x4x56x56x16xf32>
  %3 = linalg.generic {
    indexing_maps = [#map0, #map1],
    iterator_types = ["parallel", "parallel", "parallel", "parallel", "parallel"]
  } ins(%1 : tensor<1x56x56x64xf32>) outs(%2 : tensor<1x4x56x56x16xf32>) attrs = {reorder} {
  ^bb0(%arg2: f32, %arg3: f32):  // no predecessors
    linalg.yield %arg2 : f32
  } -> tensor<1x4x56x56x16xf32>

  return %3 : tensor<1x4x56x56x16xf32>
}

//       CHECK: func @reorder_fold
//  CHECK-SAME:   %[[arg0:.*]]: tensor<1x4x56x56x16xf32>
//  CHECK-NEXT:   return %[[arg0]] : tensor<1x4x56x56x16xf32>
