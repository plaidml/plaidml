// RUN: pmlc-opt %s | pmlc-opt | FileCheck %s

// CHECK-LABEL: @layer
func @layer(%arg0: tensor<10x20xf32>) -> tensor<10x20xf32> {
  // CHECK: layer.box "foo" (%{{.*}}) = (%{{.*}}) : (tensor<10x20xf32>) -> tensor<10x20xf32>
  %0 = layer.box "foo" (%arg1) = (%arg0) : (tensor<10x20xf32>) -> tensor<10x20xf32> {
    // CHECK: layer.return %{{.*}} : tensor<10x20xf32>
    layer.return %arg1 : tensor<10x20xf32>
  }
  return %0 : tensor<10x20xf32>
}
