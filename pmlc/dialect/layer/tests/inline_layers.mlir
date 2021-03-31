// RUN: pmlc-opt -layer-inline-layers %s | FileCheck %s

func @relu(%arg0: tensor<10x20xf32>) -> tensor<10x20xf32> {
  %0 = layer.box "relu" (%arg1) = (%arg0) : (tensor<10x20xf32>) -> tensor<10x20xf32> {
    %cst = tile.constant(0.000000e+00 : f64) : tensor<f32>
    %1 = tile.cmp_lt %arg1, %cst : (tensor<10x20xf32>, tensor<f32>) -> tensor<10x20xi1>
    %2 = tile.select %1, %cst, %arg1 : (tensor<10x20xi1>, tensor<f32>, tensor<10x20xf32>) -> tensor<10x20xf32>
    layer.return %2 : tensor<10x20xf32>
  }
  return %0 : tensor<10x20xf32>
}

// CHECK-LABEL: func @relu
// CHECK-NOT:     layer.box
// CHECK:         tile.constant
// CHECK:         tile.cmp_lt
// CHECK:         tile.select
// CHECK-NOT:     layer.return
// CHECK:         return
