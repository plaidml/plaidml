// RUN: pmlc-opt -tile-compute-bounds -convert-tile-to-linalg %s | FileCheck %s

#map0 = affine_map<(d0, d1, d2) -> (d0, d1)>
#map1 = affine_map<(d0, d1, d2) -> (d0, d2)>
#map2 = affine_map<(d0, d1, d2) -> (d2, d1)>
func @matrixPower(%m : tensor<16x16xf32>) -> tensor<16x16xf32> {
  %fzero = tile.constant(0.0 : f32) : tensor<16x16xf32>
  %zero = constant 0 : index
  %one = constant 1 : index
  %four = constant 4 : index
  %out = scf.for %i = %zero to %four step %one iter_args (%cur = %m) -> tensor<16x16xf32> {
    %next = tile.contract add, mul, %fzero, %cur, %m {sink=#map0, srcs=[#map1, #map2]} : tensor<16x16xf32>, tensor<16x16xf32>, tensor<16x16xf32> -> tensor<16x16xf32>
    scf.yield %next : tensor<16x16xf32>
  }
  return %out : tensor<16x16xf32>
}

// CHECK-LABEL: @matrixPower
// CHECK-SAME: (%[[arg0:.*]]: tensor<16x16xf32>) -> tensor<16x16xf32>
// CHECK: %[[cst:.*]] = constant 0.000000e+00
// CHECK: %[[c0:.*]] = constant 0
// CHECK: %[[c1:.*]] = constant 1
// CHECK: %[[c4:.*]] = constant 4
// CHECK: scf.for {{.*}} = %[[c0]] to %[[c4]] step %[[c1]] iter_args(%[[arg2:.*]] = %[[arg0]])
// CHECK:   %[[t1:.*]] = linalg.init_tensor [16, 16] : tensor<16x16xf32>
// CHECK:   %[[t2:.*]] = linalg.fill(%[[cst]], %[[t1]]) : f32, tensor<16x16xf32> -> tensor<16x16xf32> 
// CHECK:   %[[t3:.*]] = linalg.generic
// CHECK-SAME: iterator_types = ["parallel", "parallel", "reduction"]}
// CHECK-SAME: ins(%[[arg2]], %[[arg0]] : tensor<16x16xf32>, tensor<16x16xf32>) outs(%[[t2]] : tensor<16x16xf32>)
// CHECK:      ^bb0(%[[arg3:.*]]: f32, %[[arg4:.*]]: f32, %[[arg5:.*]]: f32):
// CHECK:        %[[t4:.*]] = arith.mulf %[[arg3]], %[[arg4]] : f32
// CHECK:        %[[t5:.*]] = arith.addf %[[arg5]], %[[t4]] : f32
// CHECK:        linalg.yield %[[t5]] : f32
// CHECK:   scf.yield %[[t3]]
// CHECK: return
