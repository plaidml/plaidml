// RUN: pmlc-opt -tile-legalize-to-pxa -canonicalize -cse %s | FileCheck %s

#map0 = (i, j, k) -> (j, k)
#map1 = (i, j, k) -> (j, i)
#map2 = (i, j, k) -> (i, k)

!fp32 = type !eltwise.fp32
!i32 = type !eltwise.i32
func @dot(%arg0: tensor<1x784x!eltwise.fp32>, %arg1: tensor<784x512x!eltwise.fp32>) -> tensor<1x512x!eltwise.fp32> {
  %c0 = "eltwise.sconst"() {value = 0.0 : f64} : () -> !fp32
  %0 = tile.affine_const 512
  %1 = tile.affine_const 1
  %2 = tile.cion add, mul, %c0, %arg0, %arg1 {sink=#map0, srcs=[#map1, #map2]} :
    !fp32, tensor<1x784x!eltwise.fp32>, tensor<784x512x!eltwise.fp32> -> tensor<1x512x!eltwise.fp32>
  return %2 : tensor<1x512x!eltwise.fp32>
}

// CHECK-LABEL: func @dot
// CHECK-SAME: %arg0: memref<1x784xf32>, %arg1: memref<784x512xf32>, %arg2: memref<1x512xf32>
// CHECK: pxa.parallel_for
// CHECK: ^bb0(%arg3: index, %arg4: index, %arg5: index):
// CHECK:   %0 = affine.load %arg0[%arg4, %arg3] : memref<1x784xf32>
// CHECK:   %1 = affine.load %arg1[%arg3, %arg5] : memref<784x512xf32>
// CHECK:   %2 = mulf %0, %1 : f32
// CHECK:   "pxa.reduce"(%2, %arg2, %arg3, %arg4, %arg5) {agg = 1 : i64, map = #map2} : (f32, memref<1x512xf32>, index, index, index) -> ()
// CHECK: ranges = [784, 1, 512]
