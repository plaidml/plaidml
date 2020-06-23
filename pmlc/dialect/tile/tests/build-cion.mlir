// RUN: pmlc-opt -tile-make-program -canonicalize -cse -split-input-file %s | FileCheck %s

// CHECK: #[[MAP0:map[0-9]+]] = affine_map<(d0, d1, d2) -> (d0, d1)>
// CHECK: #[[MAP1:map[0-9]+]] = affine_map<(d0, d1, d2) -> (d0, d2)>
// CHECK: #[[MAP2:map[0-9]+]] = affine_map<(d0, d1, d2) -> (d2, d1)>
func @dot(%arg0: tensor<1x2xf32>, %arg1: tensor<2x3xf32>) -> tensor<?x?xf32> {
  %c0 = "eltwise.sconst"() {value = 0.0 : f32} : () -> tensor<f32>
  %0 = tile.idx 0
  %1 = tile.idx 1
  %2 = tile.idx 2
  %3 = tile.dim %arg0[0] : tensor<1x2xf32>
  %4 = tile.dim %arg1[1] : tensor<2x3xf32>
  %5 = tile.tmap %arg0[%0, %2] : tensor<1x2xf32>
  %6 = tile.tmap %arg1[%2, %1] : tensor<2x3xf32>
  %7 = tile.map %3, %4
  %8 = tile.map %0, %1
  %9 = tile.cons ()
  %10 = tile.sym_contract add, mul, %c0, %9, %7, %8, %5, %6 : tensor<f32> -> tensor<?x?xf32>
  return %10 : tensor<?x?xf32>
}

// CHECK: func @dot(%{{.*}}: tensor<1x2xf32>, %{{.*}}: tensor<2x3xf32>) -> tensor<1x3xf32>
// CHECK: %[[CST:.*]] = "eltwise.sconst"() {value = 0.000000e+00 : f32} : () -> tensor<f32>
// CHECK: %[[CION:.*]] = tile.contract add, mul, %[[CST]], %{{.*}}, %{{.*}} {sink = #[[MAP0]], srcs = [#[[MAP1]], #[[MAP2]]]}
// CHECK-SAME: tensor<f32>, tensor<1x2xf32>, tensor<2x3xf32> -> tensor<1x3xf32>
// CHECK: return %[[CION]] : tensor<1x3xf32>

// -----

func @dot_partial_size(%arg0: tensor<?x2xf32>, %arg1: tensor<2x3xf32>) -> tensor<?x?xf32> {
  %c0 = "eltwise.sconst"() {value = 0.0 : f32} : () -> tensor<f32>
  %0 = tile.idx 0
  %1 = tile.idx 1
  %2 = tile.idx 2
  %3 = tile.dim %arg0[0] : tensor<?x2xf32>
  %4 = tile.dim %arg1[1] : tensor<2x3xf32>
  %5 = tile.tmap %arg0[%0, %2] : tensor<?x2xf32>
  %6 = tile.tmap %arg1[%2, %1] : tensor<2x3xf32>
  %7 = tile.map %3, %4
  %8 = tile.map %0, %1
  %9 = tile.cons ()
  %10 = tile.sym_contract add, mul, %c0, %9, %7, %8, %5, %6 : tensor<f32> -> tensor<?x?xf32>
  return %10 : tensor<?x?xf32>
}

// CHECK: func @dot_partial_size(%{{.*}}: tensor<?x2xf32>, %{{.*}}: tensor<2x3xf32>) -> tensor<?x3xf32> {
// CHECK:   %{{.*}} = tile.sym_contract add, mul, %{{.*}}, %{{.*}}, %{{.*}}, %{{.*}}, %{{.*}}, %{{.*}} : tensor<f32> -> tensor<?x3xf32>

// -----

func @max_axis_partial_size(%arg0: tensor<?x3x4xf32>) -> tensor<?x?xf32> {
  %c0 = "eltwise.sconst"() {value = 0.0 : f32} : () -> tensor<f32>
  %0 = tile.idx 0
  %1 = tile.idx 1
  %2 = tile.idx 2
  %3 = tile.dim %arg0[0] : tensor<?x3x4xf32>
  %4 = tile.dim %arg0[1] : tensor<?x3x4xf32>
  %5 = tile.dim %arg0[2] : tensor<?x3x4xf32>
  %6 = tile.tmap %arg0[%0, %1, %2] : tensor<?x3x4xf32>
  %7 = tile.map %3, %4
  %8 = tile.map %0, %1
  %9 = tile.cons ()
  %10 = tile.sym_contract max, none, %c0, %9, %7, %8, %6 : tensor<f32> -> tensor<?x?xf32>
  return %10 : tensor<?x?xf32>
}

// CHECK: func @max_axis_partial_size(%{{.*}}: tensor<?x3x4xf32>) -> tensor<?x3xf32> {
// CHECK:   %{{.*}} = tile.sym_contract max, none, %{{.*}}, %{{.*}}, %{{.*}}, %{{.*}}, %{{.*}} : tensor<f32> -> tensor<?x3xf32>

// -----

func @cumsum(%arg0: tensor<10xf32>) -> tensor<?xf32> {
  %c0 = "eltwise.sconst"() {value = 0.0 : f32} : () -> f32
  %0 = tile.idx 0
  %1 = tile.idx 1
  %2 = tile.dim %arg0[0] : tensor<10xf32> 
  %3 = tile.tmap %arg0[%0] : tensor<10xf32>
  %4 = tile.map %2
  %5 = tile.map %1
  %6 = tile.poly_sub %1, %0
  %7 = tile.cons (%6, %2)
  %8 = tile.sym_contract add, none, %c0, %7, %4, %5, %3 : f32 -> tensor<?xf32>
  return %8 : tensor<?xf32>
}

// CHECK: #[[MAP0:map[0-9]+]] = affine_map<(d0, d1) -> (d0)>
// CHECK: #[[MAP1:map[0-9]+]] = affine_map<(d0, d1) -> (d1)>
// CHECK: #[[SET0:set[0-9]+]] = affine_set<(d0, d1) : (d0 - d1 >= 0, -d0 + d1 + 9 >= 0)>

// CHECK: func @cumsum(%arg0: tensor<10xf32>) -> tensor<10xf32> {
// CHECK:   %[[CST:.*]] = "eltwise.sconst"() {value = 0.000000e+00 : f32} : () -> f32
// CHECK:   %[[CION:.*]] = tile.contract add, none, %[[CST]], %arg0
// CHECK-SAME: {cons = #[[SET0]], sink = #[[MAP0]], srcs = [#[[MAP1]]]}
// CHECK-SAME: f32, tensor<10xf32> -> tensor<10xf32>
// CHECK:   return %[[CION]] : tensor<10xf32>
// CHECK: }

// -----

// CHECK-LABEL: func @infer_cion_result
// CHECK-SAME: tensor<3xf16>) -> tensor<3xf16>
func @infer_cion_result(%arg0: tensor<3xf16>) -> tensor<3xf32> {
  %0 = tile.idx 0
  %zero = "eltwise.sconst"() {value = 0.0 : f64} : () -> tensor<f16>
  %two = "eltwise.sconst"() {value = 2.0 : f64} : () -> tensor<f16>
  %1 = "eltwise.mul"(%arg0, %two) : (tensor<3xf16>, tensor<f16>) -> tensor<3xf32>
  %2 = tile.tmap %1[%0] : tensor<3xf32>
  %3 = tile.map %0
  %c3 = tile.constant 3
  %4 = tile.map %c3
  %5 = tile.cons ()
  %6 = tile.sym_contract assign, none, %zero, %5, %4, %3, %2 : tensor<f16> -> tensor<3xf32>
  // CHECK: tile.contract assign, none, %{{.*}}, %{{.*}} {sink = #{{.*}}, srcs = [#{{.*}}]} : tensor<f16>, tensor<3xf16> -> tensor<3xf16>
  return %6 : tensor<3xf32>
  // CHECK: return %{{.*}} : tensor<3xf16>
}
