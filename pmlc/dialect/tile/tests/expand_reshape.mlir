// RUN: pmlc-opt -tile-expand-reshape %s | FileCheck %s

// CHECK: #[[$map0:.*]] = affine_map<(d0, d1, d2, d3) -> (0, d0, d1, d3 + d2 * 4)>
// CHECK: #[[$map1:.*]] = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>
// CHECK: #[[$map2:.*]] = affine_map<(d0) -> (d0, 0)>
// CHECK: #[[$map3:.*]] = affine_map<(d0) -> (0, 0, d0)>
// CHECK: #[[$map4:.*]] = affine_map<(d0, d1, d2, d3) -> (d1 + d0 * 3, 0, d2, 0, d3)>
// CHECK: #[[$map5:.*]] = affine_map<(d0, d1, d2, d3) -> (d0, 0, d1, d3 + d2 * 4)>
// CHECK: #[[$map6:.*]] = affine_map<(d0, d1) -> (d1 + d0 * 16)>
// CHECK: #[[$map7:.*]] = affine_map<(d0, d1) -> (d0, d1)>
// CHECK: #[[$map8:.*]] = affine_map<(d0, d1) -> (d1 + d0 * 12)>

// CHECK-LABEL: @reshape_4x4x4x4
func @reshape_4x4x4x4(%arg0: tensor<4x4x4x4xf16>) -> tensor<1x4x4x16xf16> {
// CHECK-SAME: (%[[arg0:.*]]: tensor<4x4x4x4xf16>) -> tensor<1x4x4x16xf16>
  %0 = tile.reshape %arg0 : (tensor<4x4x4x4xf16>) -> tensor<1x4x4x16xf16>
  // CHECK: %[[ret:.*]] = tile.contract assign, none, {{.*}}, %[[arg0]] {sink = #[[$map0]], srcs = [#[[$map1]]]} : tensor<f16>, tensor<4x4x4x4xf16> -> tensor<1x4x4x16xf16>
  return %0 : tensor<1x4x4x16xf16>
  // CHECK: return %[[ret]]
}

// CHECK-LABEL: @reshape_1x1x1000
func @reshape_1x1x1000(%arg0: tensor<1x1x1000xf16>) -> tensor<1000x1xf16> {
// CHECK-SAME: (%[[arg0:.*]]: tensor<1x1x1000xf16>) -> tensor<1000x1xf16>
  %0 = tile.reshape %arg0 : (tensor<1x1x1000xf16>) -> tensor<1000x1xf16>
  // CHECK: %[[ret:.*]] = tile.contract assign, none, {{.*}}, %[[arg0]] {sink = #[[$map2]], srcs = [#[[$map3]]]} : tensor<f16>, tensor<1x1x1000xf16> -> tensor<1000x1xf16>
  return %0 : tensor<1000x1xf16>
  // CHECK: return %[[ret]]
}

// CHECK-LABEL: @reshape_1000x1
func @reshape_1000x1(%arg0: tensor<1000x1xf16>) -> tensor<1x1x1000xf16> {
// CHECK-SAME: (%[[arg0:.*]]: tensor<1000x1xf16>) -> tensor<1x1x1000xf16>
  %0 = tile.reshape %arg0 : (tensor<1000x1xf16>) -> tensor<1x1x1000xf16>
  // CHECK: %[[ret:.*]] = tile.contract assign, none, {{.*}}, %[[arg0]] {sink = #[[$map3]], srcs = [#[[$map2]]]} : tensor<f16>, tensor<1000x1xf16> -> tensor<1x1x1000xf16>
  return %0 : tensor<1x1x1000xf16>
  // CHECK: return %[[ret]]
}

// CHECK-LABEL: @reshape_3x1x3x16
func @reshape_3x1x3x16(%arg0: tensor<3x1x3x16xf16>) -> tensor<9x1x4x1x4xf16> {
// CHECK-SAME: (%[[arg0:.*]]: tensor<3x1x3x16xf16>) -> tensor<9x1x4x1x4xf16>
  %0 = tile.reshape %arg0 : (tensor<3x1x3x16xf16>) -> tensor<9x1x4x1x4xf16>
  // CHECK: %[[ret:.*]] = tile.contract assign, none, {{.*}}, %[[arg0]] {sink = #[[$map4]], srcs = [#[[$map5]]]} : tensor<f16>, tensor<3x1x3x16xf16> -> tensor<9x1x4x1x4xf16>
  return %0 : tensor<9x1x4x1x4xf16>
  // CHECK: return %[[ret]]
}

// CHECK-LABEL: @reshape_9x16
func @reshape_9x16(%arg0: tensor<9x16xf16>) -> tensor<12x12xf16> {
// CHECK-SAME: (%[[arg0:.*]]: tensor<9x16xf16>) -> tensor<12x12xf16>
  %0 = tile.reshape %arg0 : (tensor<9x16xf16>) -> tensor<12x12xf16>
  // CHECK: %[[tmp:.*]] = tile.contract assign, none, {{.*}}, %[[arg0]] {sink = #[[$map6]], srcs = [#[[$map7]]]} : tensor<f16>, tensor<9x16xf16> -> tensor<144xf16>
  // CHECK: %[[ret:.*]] = tile.contract assign, none, {{.*}}, %[[tmp]] {sink = #[[$map7]], srcs = [#[[$map8]]]} : tensor<f16>, tensor<144xf16> -> tensor<12x12xf16>
  return %0 : tensor<12x12xf16>
  // CHECK: return %[[ret]]
}
