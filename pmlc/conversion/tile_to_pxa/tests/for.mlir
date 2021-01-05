// RUN: pmlc-opt -tile-compute-bounds -convert-tile-to-pxa %s | FileCheck %s

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
// CHECK-SAME: (%[[arg0:.*]]: memref<16x16xf32>, %[[arg1:.*]]: memref<16x16xf32>)
// CHECK: %[[cst:.*]] = constant 0.000000e+00
// CHECK: %[[c0:.*]] = constant 0
// CHECK: %[[c1:.*]] = constant 1
// CHECK: %[[c4:.*]] = constant 4
// CHECK: scf.for {{.*}} = %[[c0]] to %[[c4]] step %[[c1]] iter_args({{.*}} = %[[arg0]])
// CHECK:   alloc() : memref<16x16xf32>
// CHECK:   affine.parallel ({{.*}}, {{.*}}) = (0, 0) to (16, 16)
// CHECK:     pxa.reduce assign
// CHECK:     affine.yield
// CHECK:   %[[res:.*]] = affine.parallel ({{.*}}, {{.*}}, {{.*}}) = (0, 0, 0) to (16, 16, 16)
// CHECK:     pxa.load
// CHECK:     pxa.load
// CHECK:     mulf
// CHECK:     pxa.reduce addf
// CHECK:     affine.yield
// CHECK:   scf.yield %[[res]]
// CHECK: return
