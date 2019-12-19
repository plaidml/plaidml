// RUN: pmlc-opt -tile-legalize-to-pxa -canonicalize -cse -split-input-file %s | FileCheck %s --dump-input-on-failure

func @eltwise_add(
  %arg0: tensor<10x20x!eltwise.fp32>,
  %arg1: tensor<10x20x!eltwise.fp32>
) -> tensor<10x20x!eltwise.fp32> {
  %0 = "eltwise.add"(%arg1, %arg0) {type = !eltwise.fp32} : (
    tensor<10x20x!eltwise.fp32>,
    tensor<10x20x!eltwise.fp32>
  ) -> tensor<10x20x!eltwise.fp32>
  return %0 : tensor<10x20x!eltwise.fp32>
}

// CHECK-LABEL: func @eltwise_add

// -----

!fp32 = type !eltwise.fp32
!i32 = type !eltwise.i32
func @dot(%arg0: tensor<1x784x!eltwise.fp32>, %arg1: tensor<784x512x!eltwise.fp32>) -> tensor<1x512x!eltwise.fp32> {
  %c0 = "eltwise.sconst"() {value = 0.0 : f64} : () -> !fp32
  %0 = tile.affine_const 512
  %1 = tile.affine_const 1
  %2 = tile.cion add, mul, %c0, %arg0, %arg1 {sink=(i, j, k) -> (j, k), srcs=[(i, j, k) -> (j, i), (i, j, k) -> (i, k)]} :
    !fp32, tensor<1x784x!eltwise.fp32>, tensor<784x512x!eltwise.fp32> -> tensor<1x512x!eltwise.fp32>
  return %2 : tensor<1x512x!eltwise.fp32>
}

// CHECK-LABEL: func @dot

// -----

#map0 = (i, j, k) -> (j, k)
#map1 = (i, j, k) -> (j, i)
#map2 = (i, j, k) -> (i, k)

!fp32 = type !eltwise.fp32
!i32 = type !eltwise.i32
func @double_dot(
  %arg0: tensor<10x20x!eltwise.fp32>,
  %arg1: tensor<20x30x!eltwise.fp32>,
  %arg2: tensor<30x40x!eltwise.fp32>
) -> tensor<10x40x!eltwise.fp32> {
  %cst = "eltwise.sconst"() {value = 0.0 : f64} : () -> !fp32
  %0 = tile.cion add, mul, %cst, %arg0, %arg1 {sink = #map0, srcs = [#map1, #map2]} :
    !fp32, tensor<10x20x!eltwise.fp32>, tensor<20x30x!eltwise.fp32> -> tensor<10x30x!eltwise.fp32>
  %1 = tile.cion add, mul, %cst, %0, %arg2 {sink = #map0, srcs = [#map1, #map2]} :
    !fp32, tensor<10x30x!eltwise.fp32>, tensor<30x40x!eltwise.fp32> -> tensor<10x40x!eltwise.fp32>
  return %1 : tensor<10x40x!eltwise.fp32>
}

// CHECK-LABEL: func @double_dot

// -----

!fp32 = type tensor<!eltwise.fp32>
!t_10x20xfp32 = type tensor<10x20x!eltwise.fp32>
!t_10x20xbool = type tensor<10x20x!eltwise.bool>

func @relu(%arg0: !t_10x20xfp32) -> !t_10x20xfp32 {
  %0 = "eltwise.sconst"() {value = 0.0 : f32} : () -> !fp32
  %1 = "eltwise.cmp_lt"(%arg0, %0) {type = !eltwise.fp32} : (!t_10x20xfp32, !fp32) -> !t_10x20xbool
  %2 = "eltwise.select"(%1, %0, %arg0) {type = !eltwise.fp32} : (!t_10x20xbool, !fp32, !t_10x20xfp32) -> !t_10x20xfp32
  return %2 : !t_10x20xfp32
}

// CHECK-LABEL: func @relu

