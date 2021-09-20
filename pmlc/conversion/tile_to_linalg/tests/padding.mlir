// RUN: pmlc-opt %s -split-input-file \
// RUN:   -tile-compute-bounds \
// RUN:   -tile-pad-constraints \
// RUN:   -convert-tile-to-linalg \
// RUN:   -canonicalize \
// RUN:   -cse \
// RUN:   | FileCheck %s

#conv1dcenter = affine_map<(i, j) -> (i + j - 1)>
#first = affine_map<(i, j) -> (i)>
#jin0to3 = affine_set<(i, j) : (j >= 0, 2 - j >= 0)>

func @pad_input(%arg0: tensor<10xf32>) -> tensor<10xf32> {
  %c0 = tile.constant(0.0 : f64) : tensor<f32>
  %0 = tile.contract add, none, %c0, %arg0 {cons=#jin0to3, srcs=[#conv1dcenter], sink=#first}
    : tensor<f32>, tensor<10xf32> -> tensor<10xf32>
  return %0 : tensor<10xf32>
}

//      CHECK: #[[map0:.*]] = affine_map<(d0, d1) -> (d0, d1)>
//      CHECK: #[[map1:.*]] = affine_map<(d0, d1) -> (d0 + d1)>
//      CHECK: #[[map2:.*]] = affine_map<(d0, d1) -> (d0)>
//      CHECK: func @pad_input
//      CHECK:   linalg.pad_tensor {{.*}} low[1] high[1]
//      CHECK:     linalg.yield
//      CHECK:   tensor<10xf32> to tensor<12xf32>
//      CHECK:   linalg.generic
// CHECK-SAME:     indexing_maps = [#[[map0]], #[[map1]], #[[map2]]]
// CHECK-SAME:     iterator_types = ["parallel", "reduction"]
// CHECK-SAME:     ins({{.*}}, {{.*}} : tensor<10x3xf32>, tensor<12xf32>)
// CHECK-SAME:     outs({{.*}} : tensor<10xf32>)
//      CHECK:     addf
//      CHECK:     linalg.yield

// -----

#conv1dcenter = affine_map<(i, j) -> (i + j - 1)>
#first = affine_map<(i, j) -> (i)>
#second = affine_map<(i, j) -> (j)>

func @pad_contraction(%A: tensor<10xf32>, %B: tensor<1xf32>, %C: tensor<3xf32>) -> tensor<10xf32> {
  %c0 = tile.constant(0.0 : f64) : tensor<f32>
  %0 = tile.contract add, mul, %c0, %A, %B {srcs=[#conv1dcenter, #second], sink=#first}
    : tensor<f32>, tensor<10xf32>, tensor<1xf32> -> tensor<10xf32>
  %1 = tile.contract add, mul, %c0, %0, %C {srcs=[#conv1dcenter, #second], sink=#first}
    : tensor<f32>, tensor<10xf32>, tensor<3xf32> -> tensor<10xf32>
  return %1 : tensor<10xf32>
}

//      CHECK: #[[map0:.*]] = affine_map<(d0, d1) -> (d0, d1)>
//      CHECK: #[[map1:.*]] = affine_map<(d0, d1) -> (d0 + d1)>
//      CHECK: #[[map2:.*]] = affine_map<(d0, d1) -> (d1)>
//      CHECK: #[[map3:.*]] = affine_map<(d0, d1) -> (d0 + 2)>
//      CHECK: #[[map4:.*]] = affine_map<(d0, d1) -> (d0)>
//      CHECK: func @pad_contraction
//      CHECK:   linalg.fill
//      CHECK:   linalg.fill
//      CHECK:   linalg.generic
// CHECK-SAME:     indexing_maps = [#[[map0]], #[[map1]], #[[map2]], #[[map3]]]
// CHECK-SAME:     iterator_types = ["window", "reduction"]
// CHECK-SAME:     ins({{.*}}, {{.*}}, {{.*}} : tensor<9x1xf32>, tensor<10xf32>, tensor<1xf32>)
// CHECK-SAME:     outs({{.*}} : tensor<12xf32>)
//      CHECK:     mulf
//      CHECK:     addf
//      CHECK:     linalg.yield
//      CHECK:   linalg.fill
//      CHECK:   linalg.generic
// CHECK-SAME:     indexing_maps = [#[[map1]], #[[map2]], #[[map4]]]
// CHECK-SAME:     iterator_types = ["parallel", "reduction"]
// CHECK-SAME:     ins({{.*}}, {{.*}} : tensor<12xf32>, tensor<3xf32>)
// CHECK-SAME:     outs({{.*}} : tensor<10xf32>)
//      CHECK:     mulf
//      CHECK:     addf
//      CHECK:     linalg.yield

// -----

// #map0 = affine_map<() -> (0, 0, 0, 0, 0, 0, 0)>
// #map1 = affine_map<(d0, d1, d2, d3, d4, d5, d6) -> (d0, d1, d2, d3)>
// #map2 = affine_map<(d0, d1, d2, d3, d4, d5, d6) -> (d0, d1 + d4 - 1, d2 + d5 - 1, d6)>
// #map3 = affine_map<(d0, d1, d2, d3, d4, d5, d6) -> (d4, d5, d6, d3)>
// #map4 = affine_map<() -> (0, 55, 55, 63, 2, 2, 63)>

// func @pad_conv3x3(%arg0: tensor<1x56x56x64xf32>, %arg1: tensor<3x3x64x64xf32>) -> tensor<1x56x56x64xf32> {
//   %cst = tile.constant(0.000000e+00 : f64) : tensor<f32>
//   %0 = tile.ident %arg0 {padLower = [0, 1, 1, 0], padType = 1 : i64, padUpper = [0, 1, 1, 0]} : (tensor<1x56x56x64xf32>) -> tensor<1x56x56x64xf32>
//   %1 = tile.contract add, mul, %cst, %0, %arg1 {lowerBounds = #map0, sink = #map1, srcs = [#map2, #map3], upperBounds = #map4} : tensor<f32>, tensor<1x56x56x64xf32>, tensor<3x3x64x64xf32> -> tensor<1x56x56x64xf32>
//   return %1 : tensor<1x56x56x64xf32>
// }
