// RUN: pmlc-opt -tile-compute-bounds -convert-tile-to-linalg -cse %s | FileCheck %s

#map0 = affine_map<(d0, d1, d2) -> (d0, d1)>
#map1 = affine_map<(d0, d1, d2) -> (d0, d2)>
#map2 = affine_map<(d0, d1, d2) -> (d2, d1)>
func @doubleFor(%m : tensor<16x16xf32>) -> tensor<16x16xf32> {
  %fzero = tile.constant(0.0 : f32) : tensor<16x16xf32>
  %zero = arith.constant 0 : index
  %one = arith.constant 1 : index
  %four = arith.constant 4 : index
  %outer = scf.for %i = %zero to %four step %one iter_args (%a0 = %m) -> tensor<16x16xf32> {
    %inner = scf.for %j = %zero to %four step %one iter_args (%a1 = %a0) -> tensor<16x16xf32> {
      %next = tile.contract add, mul, %fzero, %a1, %m {sink=#map0, srcs=[#map1, #map2]} : tensor<16x16xf32>, tensor<16x16xf32>, tensor<16x16xf32> -> tensor<16x16xf32>
      scf.yield %next : tensor<16x16xf32>
    }
    scf.yield %inner : tensor<16x16xf32>
  }
  return %outer : tensor<16x16xf32>
}

// CHECK-LABEL: @doubleFor
// CHECK-SAME: (%[[arg0:.*]]: tensor<16x16xf32>) -> tensor<16x16xf32>
// CHECK: %[[cst:.*]] = arith.constant 0.000000e+00
// CHECK: %[[c0:.*]] = arith.constant 0
// CHECK: %[[c1:.*]] = arith.constant 1
// CHECK: %[[c4:.*]] = arith.constant 4
// CHECK: scf.for {{.*}} = %[[c0]] to %[[c4]] step %[[c1]] iter_args(%[[arg3:.*]] = %[[arg0]])
// CHECK:   %[[r0:.*]] = scf.for {{.*}} = %[[c0]] to %[[c4]] step %[[c1]] iter_args(%[[arg5:.*]] = %[[arg3]])
// CHECK:     linalg.init_tensor [16, 16] : tensor<16x16xf32>
// CHECK:     linalg.fill ins(%[[cst]] : f32) outs({{.*}} : tensor<16x16xf32>) -> tensor<16x16xf32>
// CHECK:     %[[r1:.*]] = linalg.generic
// CHECK:       mulf
// CHECK:       addf
// CHECK:       linalg.yield
// CHECK:     scf.yield %[[r1]]
// CHECK:   scf.yield %[[r0]]
// CHECK: return
