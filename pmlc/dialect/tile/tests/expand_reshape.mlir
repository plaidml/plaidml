// RUN: pmlc-opt -expand-reshape %s | FileCheck %s

// CHECK: #map0 = affine_map<(d0, d1, d2, d3) -> (0, d0, d1, d3 + d2 * 4)>
// CHECK: #map1 = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>
// CHECK: @reshape_4x4x4x4
func @reshape_4x4x4x4(%arg0: tensor<4x4x4x4xf16>) -> tensor<1x4x4x16xf16> {
// CHECK-SAME: (%[[arg0:.*]]: tensor<4x4x4x4xf16>) -> tensor<1x4x4x16xf16>
  %0 = tile.reshape %arg0 : (tensor<4x4x4x4xf16>) -> tensor<1x4x4x16xf16>
  // CHECK: %[[ret:.*]] = tile.contract assign, none, {{.*}}, %[[arg0]] {sink = #map0, srcs = [#map1]} : tensor<f16>, tensor<4x4x4x4xf16> -> tensor<1x4x4x16xf16>
  return %0 : tensor<1x4x4x16xf16>
  // CHECK: return %[[ret]]
}
