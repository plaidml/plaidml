// RUN: pmlc-opt -tile-materialize %s | FileCheck %s

func @relu(%arg0: tensor<10x20xf32>) -> tensor<10x20xf32> {
  %cst = tile.constant(0.0 : f64) : tensor<!tile.fx>
  %0 = tile.cmp_lt %arg0, %cst : (tensor<10x20xf32>, tensor<!tile.fx>) -> tensor<10x20xi1>
  %1 = tile.cast %cst : (tensor<!tile.fx>) -> tensor<f32>
  %2 = tile.select %0, %1, %arg0 : (tensor<10x20xi1>, tensor<f32>, tensor<10x20xf32>) -> tensor<10x20xf32>
  return %2 : tensor<10x20xf32>
}

// CHECK-LABEL: func @relu
// CHECK: tile.constant
// CHECK: tile.cast %{{.*}} : (tensor<!tile.fx>) -> tensor<f32>
// CHECK: tile.cmp_lt %{{.*}}, %{{.*}} : (tensor<10x20xf32>, tensor<f32>) -> tensor<10x20xi1>
