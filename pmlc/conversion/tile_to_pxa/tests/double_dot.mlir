// RUN: pmlc-opt -tile-compute-bounds -convert-tile-to-pxa -canonicalize -cse %s | FileCheck %s

#map0 = affine_map<(i, j, k) -> (j, k)>
#map1 = affine_map<(i, j, k) -> (j, i)>
#map2 = affine_map<(i, j, k) -> (i, k)>

func @double_dot(
  %arg0: tensor<10x20xf32>,
  %arg1: tensor<20x30xf32>,
  %arg2: tensor<30x40xf32>
) -> tensor<10x40xf32> {
  %cst = "eltwise.sconst"() {value = 0.0 : f64} : () -> f32
  %0 = tile.contract add, mul, %cst, %arg0, %arg1 {sink = #map0, srcs = [#map1, #map2]} :
    f32, tensor<10x20xf32>, tensor<20x30xf32> -> tensor<10x30xf32>
  %1 = tile.contract add, mul, %cst, %0, %arg2 {sink = #map0, srcs = [#map1, #map2]} :
    f32, tensor<10x30xf32>, tensor<30x40xf32> -> tensor<10x40xf32>
  return %1 : tensor<10x40xf32>
}

// CHECK-LABEL: func @double_dot
// CHECK-SAME: %{{.*}}: memref<10x20xf32>
// CHECK-SAME: %{{.*}}: memref<20x30xf32>
// CHECK-SAME: %{{.*}}: memref<30x40xf32>
// CHECK-SAME: -> memref<10x40xf32>
// CHECK: affine.parallel (%{{.*}}, %{{.*}}, %{{.*}}) = (0, 0, 0) to (20, 10, 30)
// CHECK:   pxa.load %{{.*}}[%{{.*}}, %{{.*}}] : memref<10x20xf32>
// CHECK:   pxa.load %{{.*}}[%{{.*}}, %{{.*}}] : memref<20x30xf32>
// CHECK:   mulf %{{.*}}, %{{.*}} : f32
// CHECK:   pxa.reduce addf %{{.*}}, %{{.*}}[%{{.*}}, %{{.*}}] : memref<10x30xf32>
// CHECK: affine.parallel (%{{.*}}, %{{.*}}, %{{.*}}) = (0, 0, 0) to (30, 10, 40)
// CHECK:   pxa.load %{{.*}}[%{{.*}}, %{{.*}}] : memref<10x30xf32>
// CHECK:   pxa.load %{{.*}}[%{{.*}}, %{{.*}}] : memref<30x40xf32>
// CHECK:   mulf %{{.*}}, %{{.*}} : f32
// CHECK:   pxa.reduce addf %{{.*}}, %{{.*}}[%{{.*}}, %{{.*}}] : memref<10x40xf32>
