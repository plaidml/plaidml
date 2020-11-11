// RUN: pmlc-opt %s | pmlc-opt | FileCheck %s

// CHECK-LABEL: @layer
func @layer(%arg0: tensor<10x20xf32>) -> tensor<10x20xf32> {
  // CHECK: tile.layer "foo" (%{{.*}}) = (%{{.*}}) : (tensor<10x20xf32>) -> tensor<10x20xf32>
  %0 = tile.layer "foo" (%arg1) = (%arg0) : (tensor<10x20xf32>) -> tensor<10x20xf32> {
    // CHECK: tile.layer.return %{{.*}} : tensor<10x20xf32>
    tile.layer.return %arg1 : tensor<10x20xf32>
  }
  return %0 : tensor<10x20xf32>
}
