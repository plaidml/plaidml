// RUN: pmlc-opt -tile-compute-bounds -convert-tile-to-pxa -canonicalize -cse %s | FileCheck %s

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
// CHECK-SAME: %arg0: memref<10x20xf32>, %arg1: memref<20x30xf32>, %arg2: memref<30x40xf32>, %arg3: memref<10x40xf32>
// CHECK: %0 = alloc() : memref<10x30xf32>
// CHECK: pxa.parallel_for
// CHECK: ^bb0(%arg4: index, %arg5: index, %arg6: index):
// CHECK:   %1 = affine.load %arg0[%arg5, %arg4] : memref<10x20xf32>
// CHECK:   %2 = affine.load %arg1[%arg4, %arg6] : memref<20x30xf32>
// CHECK:   %3 = mulf %1, %2 : f32
// CHECK:   "pxa.reduce"(%3, %0, %arg4, %arg5, %arg6) {agg = 1 : i64, map = #map2} : (f32, memref<10x30xf32>, index, index, index) -> ()
// CHECK: ranges = [20, 10, 30]
// CHECK: pxa.parallel_for
// CHECK: ^bb0(%arg4: index, %arg5: index, %arg6: index):
// CHECK:   %1 = affine.load %0[%arg5, %arg4] : memref<10x30xf32>
// CHECK:   %2 = affine.load %arg2[%arg4, %arg6] : memref<30x40xf32>
// CHECK:   %3 = mulf %1, %2 : f32
// CHECK:   "pxa.reduce"(%3, %arg3, %arg4, %arg5, %arg6) {agg = 1 : i64, map = #map2} : (f32, memref<10x40xf32>, index, index, index) -> ()
// CHECK: ranges = [30, 10, 40]
