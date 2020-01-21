// RUN: pmlc-opt -tile-compute-bounds -convert-tile-to-pxa -canonicalize -cse %s | FileCheck %s

#map0 = affine_map<(i, j, k) -> (j, k)>
#map1 = affine_map<(i, j, k) -> (j, i)>
#map2 = affine_map<(i, j, k) -> (i, k)>

!f32 = type !eltwise.f32
!i32 = type !eltwise.i32
func @dot(%arg0: tensor<1x784x!eltwise.f32>, %arg1: tensor<784x512x!eltwise.f32>) -> tensor<1x512x!eltwise.f32> {
  %c0 = "eltwise.sconst"() {value = 0.0 : f64} : () -> !f32
  %0 = tile.affine_const 512
  %1 = tile.affine_const 1
  %2 = tile.cion add, mul, %c0, %arg0, %arg1 {sink=#map0, srcs=[#map1, #map2]} :
    !f32, tensor<1x784x!eltwise.f32>, tensor<784x512x!eltwise.f32> -> tensor<1x512x!eltwise.f32>
  return %2 : tensor<1x512x!eltwise.f32>
}

// CHECK-DAG: [[map_dot:#map[0-9]+]] = affine_map<() -> (784, 1, 512)>
// CHECK-LABEL: func @dot
// CHECK-SAME: %{{.*}}: memref<1x784xf32>, %{{.*}}: memref<784x512xf32>, %{{.*}}: memref<1x512xf32>
// CHECK: pxa.parallel_for
// CHECK: ^bb0(%{{.*}}: index, %{{.*}}: index, %{{.*}}: index):
// CHECK:   affine.load %{{.*}}[%{{.*}}, %{{.*}}] : memref<1x784xf32>
// CHECK:   affine.load %{{.*}}[%{{.*}}, %{{.*}}] : memref<784x512xf32>
// CHECK:   mulf %{{.*}}, %{{.*}} : f32
// CHECK:   pxa.reduce add %{{.*}}, %{{.*}}[%{{.*}}, %{{.*}}] : memref<1x512xf32>
// CHECK: ranges = [[map_dot]]
