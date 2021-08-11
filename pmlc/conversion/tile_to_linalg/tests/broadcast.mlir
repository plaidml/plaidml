// RUN: pmlc-opt -convert-tile-to-linalg -canonicalize -cse %s | FileCheck %s

// CHECK: #[[map0:.*]] = affine_map<(d0, d1, d2, d3) -> (0, d1, d2, d3)>
// CHECK: #[[map1:.*]] = affine_map<(d0, d1, d2, d3) -> (d0, 0, d2, d3)>
// CHECK: #[[map2:.*]] = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>
// CHECK: #[[map3:.*]] = affine_map<(d0, d1) -> ()>
// CHECK: #[[map4:.*]] = affine_map<(d0, d1) -> (d0, d1)>

// CHECK: func @broadcast_has_dim_one
func @broadcast_has_dim_one(%arg0: tensor<1x2x3x4xui64>, %arg1: tensor<2x1x3x4xui64>) -> tensor<2x2x3x4xi1> {
  %0 = tile.cmp_ge %arg0, %arg1 : (tensor<1x2x3x4xui64>, tensor<2x1x3x4xui64>) -> tensor<2x2x3x4xi1>
  return %0 : tensor<2x2x3x4xi1>

  // CHECK: linalg.init_tensor [2, 2, 3, 4]
  // CHECK: linalg.generic
  // CHECK-SAME:   indexing_maps = [#[[map0]], #[[map1]], #[[map2]]]
  // CHECK-SAME:   ins(%{{.*}}, %{{.*}} : tensor<1x2x3x4xi64>, tensor<2x1x3x4xi64>) outs(%{{.*}} : tensor<2x2x3x4xi1>)
  // CHECK:    cmpi uge
  // CHECK:    linalg.yield
}

// CHECK: func @broadcast_matrix_scalar
func @broadcast_matrix_scalar(%arg0: tensor<ui64>, %arg1: tensor<3x4xui64>) -> tensor<3x4xi1> {
  %0 = tile.cmp_ge %arg0, %arg1 : (tensor<ui64>, tensor<3x4xui64>) -> tensor<3x4xi1>
  return %0 : tensor<3x4xi1>

  // CHECK: linalg.init_tensor [3, 4]
  // CHECK: linalg.generic
  // CHECK-SAME:   indexing_maps = [#[[map3]], #[[map4]], #[[map4]]]
  // CHECK-SAME:   ins(%{{.*}}, %{{.*}} : tensor<i64>, tensor<3x4xi64>) outs(%{{.*}} : tensor<3x4xi1>)
  // CHECK:    cmpi uge
  // CHECK:    linalg.yield
}
