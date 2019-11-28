// RUN: pmlc-opt -canonicalize -cse -split-input-file %s | FileCheck %s

!fp32 = type tensor<!eltwise.fp32>

func @dot(%arg0: tensor<1x2x!eltwise.fp32>, %arg1: tensor<2x3x!eltwise.fp32>) -> tensor<?x?x!eltwise.fp32> {
  %c0 = "eltwise.sconst"() {value = 0.0 : f32} : () -> !fp32
  %c3 = "tile.affine_const"() {value = 3 : i64} : () -> index
  %c1 = "tile.affine_const"() {value = 1 : i64} : () -> index
  %0 = "tile.idx"() : () -> index
  %1 = "tile.idx"() : () -> index
  %2 = "tile.idx"() : () -> index
  %3 = "tile.dim"(%arg0) {dim = 0 : i64} : (tensor<1x2x!eltwise.fp32>) -> index
  %4 = "tile.dim"(%arg0) {dim = 1 : i64} : (tensor<1x2x!eltwise.fp32>) -> index
  %5 = "tile.dim"(%arg1) {dim = 0 : i64} : (tensor<2x3x!eltwise.fp32>) -> index
  %6 = "tile.dim"(%arg1) {dim = 1 : i64} : (tensor<2x3x!eltwise.fp32>) -> index
  %7 = "tile.tmap"(%arg0, %0, %2) : (tensor<1x2x!eltwise.fp32>, index, index) -> !tile.tmap
  %8 = "tile.tmap"(%arg1, %2, %1) : (tensor<2x3x!eltwise.fp32>, index, index) -> !tile.tmap
  %9 = "tile.map"(%3, %6) : (index, index) -> !tile.map
  %10 = "tile.map"(%0, %1) : (index, index) -> !tile.map
  %11 = "tile.cons"() : () -> !tile.cons
  %12 = "tile.dyn_cion"(%c0, %11, %9, %10, %7, %8) { agg = 1, combo = 4 } :
    (!fp32, !tile.cons, !tile.map, !tile.map, !tile.tmap, !tile.tmap) -> tensor<?x?x!eltwise.fp32>
  return %12 : tensor<?x?x!eltwise.fp32>
}

// CHECK: #[[MAP0:map[0-9]+]] = (d0, d1, d2) -> (d0, d1)
// CHECK: #[[MAP1:map[0-9]+]] = (d0, d1, d2) -> (d0, d2)
// CHECK: #[[MAP2:map[0-9]+]] = (d0, d1, d2) -> (d2, d1)
// CHECK: #[[SET0:set[0-9]+]] = (d0, d1, d2) : (1 == 0)

// CHECK: func @dot(%arg0: tensor<1x2x!eltwise.fp32>, %arg1: tensor<2x3x!eltwise.fp32>) -> tensor<1x3x!eltwise.fp32> {
// CHECK:   %[[CST:.*]] = "eltwise.sconst"() {value = 0.000000e+00 : f32} : () -> !fp32
// CHECK:   %[[CION:.*]] = "tile.cion"(%[[CST]], %arg0, %arg1) {agg = 1 : i64, combo = 4 : i64, cons = #[[SET0]], sink = #[[MAP0]], srcs = [#[[MAP1]], #[[MAP2]]]} : (!fp32, tensor<1x2x!eltwise.fp32>, tensor<2x3x!eltwise.fp32>) -> tensor<1x3x!eltwise.fp32>
// CHECK:   return %[[CION]] : tensor<1x3x!eltwise.fp32>
// CHECK: }

// -----

!fp32 = type tensor<!eltwise.fp32>

func @cumsum(%arg0: tensor<10x!eltwise.fp32>) -> tensor<?x!eltwise.fp32> {
  %c0 = "eltwise.sconst"() {value = 0.0 : f32} : () -> !fp32
  %0 = "tile.idx"() : () -> index // %arg1 (i)
  %1 = "tile.idx"() : () -> index // %arg2 (j)
  %2 = "tile.dim"(%arg0) {dim = 0 : i64} : (tensor<10x!eltwise.fp32>) -> index
  %3 = "tile.tmap"(%arg0, %0) : (tensor<10x!eltwise.fp32>, index) -> !tile.tmap
  %4 = "tile.map"(%2) : (index) -> !tile.map // size_map
  %5 = "tile.map"(%1) : (index) -> !tile.map // sink_idx_map
  %6 = "tile.affine_sub"(%1, %0) : (index, index) -> index
  %7 = "tile.cons"(%6, %2) : (index, index) -> !tile.cons
  %8 = "tile.dyn_cion"(%c0, %7, %4, %5, %3) { agg = 1, combo = 0 } :
    (!fp32, !tile.cons, !tile.map, !tile.map, !tile.tmap) -> tensor<?x!eltwise.fp32>
  return %8 : tensor<?x!eltwise.fp32>
}

// CHECK: #[[MAP0:map[0-9]+]] = (d0, d1) -> (d0)
// CHECK: #[[MAP1:map[0-9]+]] = (d0, d1) -> (d1)
// CHECK: #[[SET0:set[0-9]+]] = (d0, d1) : (d0 - d1 - 10 >= 0)

// CHECK: func @cumsum(%arg0: tensor<10x!eltwise.fp32>) -> tensor<10x!eltwise.fp32> {
// CHECK:   %[[CST:.*]] = "eltwise.sconst"() {value = 0.000000e+00 : f32} : () -> !fp32
// CHECK:   %[[CION:.*]] = "tile.cion"(%[[CST]], %arg0) {agg = 1 : i64, combo = 0 : i64, cons = #[[SET0]], sink = #[[MAP0]], srcs = [#[[MAP1]]]} : (!fp32, tensor<10x!eltwise.fp32>) -> tensor<10x!eltwise.fp32>
// CHECK:   return %[[CION]] : tensor<10x!eltwise.fp32>
// CHECK: }
