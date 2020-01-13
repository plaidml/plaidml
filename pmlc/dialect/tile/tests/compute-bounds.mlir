// RUN: pmlc-opt -tile-compute-bounds -cse -split-input-file %s | FileCheck %s

!fp32 = type !eltwise.fp32

#map0 = (i, j, k) -> (j, k)
#map1 = (i, j, k) -> (j, i)
#map2 = (i, j, k) -> (i, k)

func @dot(%arg0: tensor<1x784x!eltwise.fp32>, %arg1: tensor<784x512x!eltwise.fp32>) -> tensor<1x512x!eltwise.fp32> {
  %c0 = "eltwise.sconst"() {value = 0.0 : f64} : () -> !fp32
  %0 = tile.affine_const 512
  %1 = tile.affine_const 1
  %2 = tile.cion add, mul, %c0, %arg0, %arg1 {sink=#map0, srcs=[#map1, #map2]} :
    !fp32, tensor<1x784x!eltwise.fp32>, tensor<784x512x!eltwise.fp32> -> tensor<1x512x!eltwise.fp32>
  return %2 : tensor<1x512x!eltwise.fp32>
}

// CHECK: #map0 = (d0, d1, d2) -> (d1, d2)
// CHECK: #map1 = (d0, d1, d2) -> (d1, d0)
// CHECK: #map2 = (d0, d1, d2) -> (d0, d2)
// CHECK-LABEL: func @dot
// CHECK: tile.cion
// CHECK-SAME: lower_bounds = [0 : index, 0 : index, 0 : index]
// CHECK-SAME: sink = #map0
// CHECK-SAME: srcs = [#map1, #map2]
// CHECK-SAME: upper_bounds = [783 : index, 0 : index, 511 : index]

// -----
