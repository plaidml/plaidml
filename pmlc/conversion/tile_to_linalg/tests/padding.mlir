// RUN: pmlc-opt -tile-compute-bounds -tile-pad-constraints -convert-tile-to-linalg -canonicalize -cse -split-input-file %s | FileCheck %s

#conv1dcenter = affine_map<(i, j) -> (i + j - 1)>
#first = affine_map<(i, j) -> (i)>
#jin0to3 = affine_set<(i, j) : (j >= 0, 2 - j >= 0)>

func @pad_input(%arg0: tensor<10xf32>) -> tensor<10xf32> {
  %c0 = tile.constant(0.0 : f64) : tensor<f32>
  %0 = tile.contract add, none, %c0, %arg0 {cons=#jin0to3, srcs=[#conv1dcenter], sink=#first}
    : tensor<f32>, tensor<10xf32> -> tensor<10xf32>
  return %0 : tensor<10xf32>
}

// CHECK: #[[map0:.*]] = affine_map<(d0) -> (d0)>
// CHECK: #[[map1:.*]] = affine_map<(d0) -> (d0 + 1)>
// CHECK: #[[map2:.*]] = affine_map<(d0, d1) -> (d0, d1)>
// CHECK: #[[map3:.*]] = affine_map<(d0, d1) -> (d0 + d1)>
// CHECK: #[[map4:.*]] = affine_map<(d0, d1) -> (d0)>
// CHECK: func @pad_input
// CHECK: linalg.generic
// CHECK-SAME:   indexing_maps = [#[[map0]], #[[map1]]]
// CHECK-SAME:   iterator_types = ["parallel"]
// CHECK-SAME:   ins({{.*}}: tensor<10xf32>) outs({{.*}} : tensor<12xf32>)
// CHECK:    linalg.yield
// CHECK: linalg.generic
// CHECK-SAME:   indexing_maps = [#[[map2]], #[[map3]], #[[map4]]]
// CHECK-SAME:   iterator_types = ["parallel", "reduction"]
// CHECK-SAME:   ins({{.*}}, {{.*}} : tensor<10x3xf32>, tensor<12xf32>) outs(%4 : tensor<10xf32>)
// CHECK:   addf
// CHECK:   linalg.yield
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

// CHECK: #[[map0:.*]] = affine_map<(d0, d1) -> (d0, d1)>
// CHECK: #[[map1:.*]] = affine_map<(d0, d1) -> (d0 + d1)>
// CHECK: #[[map2:.*]] = affine_map<(d0, d1) -> (d1)>
// CHECK: #[[map3:.*]] = affine_map<(d0, d1) -> (d0 + 2)>
// CHECK: #[[map4:.*]] = affine_map<(d0, d1) -> (d0)>
// CHECK: func @pad_contraction
// CHECK: linalg.fill
// CHECK: linalg.fill
// CHECK: linalg.generic
// CHECK-SAME:   indexing_maps = [#[[map0]], #[[map1]], #[[map2]], #[[map3]]]
// CHECK-SAME:   iterator_types = ["window", "reduction"]
// CHECK-SAME:   ins({{.*}}, {{.*}}, {{.*}} : tensor<9x1xf32>, tensor<10xf32>, tensor<1xf32>)
// CHECK-SAME:   outs({{.*}} : tensor<12xf32>)
// CHECK:   mulf
// CHECK:   addf
// CHECK:   linalg.yield
// CHECK: linalg.fill
// CHECK: linalg.generic
// CHECK-SAME:   indexing_maps = [#[[map1]], #[[map2]], #[[map4]]]
// CHECK-SAME:   iterator_types = ["parallel", "reduction"]
// CHECK-SAME:   ins({{.*}}, {{.*}} : tensor<12xf32>, tensor<3xf32>) outs({{.*}} : tensor<10xf32>)
// CHECK:   mulf
// CHECK:   addf
// CHECK:   linalg.yield

