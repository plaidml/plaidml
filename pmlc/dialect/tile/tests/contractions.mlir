// RUN: pmlc-opt -canonicalize -cse -split-input-file %s | FileCheck %s

!fp32 = type tensor<!eltwise.fp32>

func @dot(%arg0: tensor<1x2x!eltwise.fp32>, %arg1: tensor<2x3x!eltwise.fp32>) -> tensor<?x?x!eltwise.fp32> {
  %c0 = "eltwise.sconst"() {value = 0.0 : f32} : () -> !fp32
  %0 = tile.idx
  %1 = tile.idx
  %2 = tile.idx
  %3 = tile.dim %arg0[0] : tensor<1x2x!eltwise.fp32>
  %4 = tile.dim %arg1[1] : tensor<2x3x!eltwise.fp32>
  %5 = tile.tmap %arg0[%0, %2] : tensor<1x2x!eltwise.fp32>
  %6 = tile.tmap %arg1[%2, %1] : tensor<2x3x!eltwise.fp32>
  %7 = tile.map %3, %4
  %8 = tile.map %0, %1
  %9 = tile.cons ()
  %10 = tile.sym_cion add, mul, %c0, %9, %7, %8, %5, %6 : !fp32 -> tensor<?x?x!eltwise.fp32>
  return %10 : tensor<?x?x!eltwise.fp32>
}

// CHECK: #[[MAP0:map[0-9]+]] = (d0, d1, d2) -> (d0, d1)
// CHECK: #[[MAP1:map[0-9]+]] = (d0, d1, d2) -> (d0, d2)
// CHECK: #[[MAP2:map[0-9]+]] = (d0, d1, d2) -> (d2, d1)

// CHECK: func @dot(%arg0: tensor<1x2x!eltwise.fp32>, %arg1: tensor<2x3x!eltwise.fp32>) -> tensor<1x3x!eltwise.fp32> {
// CHECK:   %[[CST:.*]] = "eltwise.sconst"() {value = 0.000000e+00 : f32} : () -> !fp32
// CHECK:   %[[CION:.*]] = tile.cion add, mul, %[[CST]], %arg0, %arg1
// CHECK-SAME: {sink = #[[MAP0]], srcs = [#[[MAP1]], #[[MAP2]]]}
// CHECK-SAME: !fp32, tensor<1x2x!eltwise.fp32>, tensor<2x3x!eltwise.fp32> -> tensor<1x3x!eltwise.fp32>
// CHECK:   return %[[CION]] : tensor<1x3x!eltwise.fp32>
// CHECK: }

// -----

!fp32 = type tensor<!eltwise.fp32>

func @cumsum(%arg0: tensor<10x!eltwise.fp32>) -> tensor<?x!eltwise.fp32> {
  %c0 = "eltwise.sconst"() {value = 0.0 : f32} : () -> !fp32
  %0 = tile.idx
  %1 = tile.idx
  %2 = tile.dim %arg0[0] : tensor<10x!eltwise.fp32> 
  %3 = tile.tmap %arg0[%0] : tensor<10x!eltwise.fp32>
  %4 = tile.map %2
  %5 = tile.map %1
  %6 = tile.affine_sub %1, %0
  %7 = tile.cons (%6, %2)
  %8 = tile.sym_cion add, none, %c0, %7, %4, %5, %3 : !fp32 -> tensor<?x!eltwise.fp32>
  return %8 : tensor<?x!eltwise.fp32>
}

// CHECK: #[[MAP0:map[0-9]+]] = (d0, d1) -> (d0)
// CHECK: #[[MAP1:map[0-9]+]] = (d0, d1) -> (d1)
// CHECK: #[[SET0:set[0-9]+]] = (d0, d1) : (-(d0 - d1) + 9 >= 0)

// CHECK: func @cumsum(%arg0: tensor<10x!eltwise.fp32>) -> tensor<10x!eltwise.fp32> {
// CHECK:   %[[CST:.*]] = "eltwise.sconst"() {value = 0.000000e+00 : f32} : () -> !fp32
// CHECK:   %[[CION:.*]] = tile.cion add, none, %[[CST]], %arg0
// CHECK-SAME: {cons = #[[SET0]], sink = #[[MAP0]], srcs = [#[[MAP1]]]}
// CHECK-SAME: !fp32, tensor<10x!eltwise.fp32> -> tensor<10x!eltwise.fp32>
// CHECK:   return %[[CION]] : tensor<10x!eltwise.fp32>
// CHECK: }
