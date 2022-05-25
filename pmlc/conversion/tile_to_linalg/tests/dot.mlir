// RUN: pmlc-opt -tile-compute-bounds -convert-tile-to-linalg %s | FileCheck %s

#map0 = affine_map<(i, j, k) -> (j, k)>
#map1 = affine_map<(i, j, k) -> (j, i)>
#map2 = affine_map<(i, j, k) -> (i, k)>

func.func @dot(%arg0: tensor<1x784xf32>, %arg1: tensor<784x512xf32>) -> tensor<1x512xf32> {
  %c0 = tile.constant(0.0 : f64) : tensor<f32>
  %2 = tile.contract add, mul, %c0, %arg0, %arg1 {sink=#map0, srcs=[#map1, #map2]} :
    tensor<f32>, tensor<1x784xf32>, tensor<784x512xf32> -> tensor<1x512xf32>
  return %2 : tensor<1x512xf32>
}

// CHECK: func.func @dot
// CHECK-SAME: (%[[arg0:.*]]: tensor<1x784xf32>, %[[arg1:.*]]: tensor<784x512xf32>) -> tensor<1x512xf32>
// CHECK: %[[cst:.*]] = arith.constant 0.000000e+00 : f32
// CHECK: %[[init:.*]] = linalg.init_tensor [1, 512] : tensor<1x512xf32>
// CHECK: %[[broadcast:.*]] = linalg.fill ins(%[[cst]] : f32) outs(%[[init]] : tensor<1x512xf32>) -> tensor<1x512xf32>
// CHECK: %[[result:.*]] = linalg.generic
// CHECK-SAME:               iterator_types = ["reduction", "parallel", "parallel"]}
// CHECK-SAME:               ins(%[[arg0]], %[[arg1]] : tensor<1x784xf32>, tensor<784x512xf32>)
// CHECK-SAME:               outs(%[[broadcast]] : tensor<1x512xf32>)
// CHECK:   ^bb0(%[[arg2:.*]]: f32, %[[arg3:.*]]: f32, %[[arg4:.*]]: f32)
// CHECK:      %[[t0:.*]] = arith.mulf %[[arg2]], %[[arg3]] : f32
// CHECK:      %[[t1:.*]] = arith.addf %[[arg4]], %[[t0]] : f32
// CHECK:      linalg.yield %[[t1]] : f32
// CHECK: return %[[result]] : tensor<1x512xf32>
