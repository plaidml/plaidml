// RUN: pmlc-opt -tile-compute-bounds -convert-tile-to-pxa -canonicalize -cse %s | FileCheck %s

#map0 = affine_map<(i, j, k) -> (j, k)>
#map1 = affine_map<(i, j, k) -> (j, i)>
#map2 = affine_map<(i, j, k) -> (i, k)>

!f32 = type !eltwise.f32
!i32 = type !eltwise.i32
func @double_dot(
  %arg0: tensor<10x20x!eltwise.f32>,
  %arg1: tensor<20x30x!eltwise.f32>,
  %arg2: tensor<30x40x!eltwise.f32>
) -> tensor<10x40x!eltwise.f32> {
  %cst = "eltwise.sconst"() {value = 0.0 : f64} : () -> !f32
  %0 = tile.cion add, mul, %cst, %arg0, %arg1 {sink = #map0, srcs = [#map1, #map2]} :
    !f32, tensor<10x20x!eltwise.f32>, tensor<20x30x!eltwise.f32> -> tensor<10x30x!eltwise.f32>
  %1 = tile.cion add, mul, %cst, %0, %arg2 {sink = #map0, srcs = [#map1, #map2]} :
    !f32, tensor<10x30x!eltwise.f32>, tensor<30x40x!eltwise.f32> -> tensor<10x40x!eltwise.f32>
  return %1 : tensor<10x40x!eltwise.f32>
}

// CHECK-DAG: [[map_dot_lb:#map[0-9]+]] = affine_map<() -> (0, 0, 0)>
// CHECK-DAG: [[map_dot_1_ub:#map[0-9]+]] = affine_map<() -> (20, 10, 30)>
// CHECK-DAG: [[map_dot_2_ub:#map[0-9]+]] = affine_map<() -> (30, 10, 40)>
// CHECK-LABEL: func @double_dot
// CHECK-SAME: %{{.*}}: memref<10x20xf32>, %{{.*}}: memref<20x30xf32>, %{{.*}}: memref<30x40xf32>, %{{.*}}: memref<10x40xf32>
// CHECK: alloc() : memref<10x30xf32>
// CHECK: pxa.parallel
// CHECK: ^bb0(%{{.*}}: index, %a{{.*}}: index, %{{.*}}: index):
// CHECK:   affine.load %{{.*}}[%{{.*}}, %{{.*}}] : memref<10x20xf32>
// CHECK:   affine.load %{{.*}}[%{{.*}}, %{{.*}}] : memref<20x30xf32>
// CHECK:   mulf %{{.*}}, %{{.*}} : f32
// CHECK:   pxa.reduce add %{{.*}}, %{{.*}}[%{{.*}}, %{{.*}}] : memref<10x30xf32>
// CHECK-DAG: lowerBoundsMap = [[map_dot_lb]]
// CHECK-DAG: upperBoundsMap = [[map_dot_1_ub]]
// CHECK-DAG: steps = [1, 1, 1]
// CHECK: pxa.parallel
// CHECK: ^bb0(%{{.*}}: index, %{{.*}}: index, %{{.*}}: index):
// CHECK:   affine.load %{{.*}}[%{{.*}}, %{{.*}}] : memref<10x30xf32>
// CHECK:   affine.load %{{.*}}[%{{.*}}, %{{.*}}] : memref<30x40xf32>
// CHECK:   mulf %{{.*}}, %{{.*}} : f32
// CHECK:   pxa.reduce add %{{.*}}, %{{.*}}[%{{.*}}, %{{.*}}] : memref<10x40xf32>
// CHECK-DAG: lowerBoundsMap = [[map_dot_lb]]
// CHECK-DAG: upperBoundsMap = [[map_dot_2_ub]]
// CHECK-DAG: steps = [1, 1, 1]
