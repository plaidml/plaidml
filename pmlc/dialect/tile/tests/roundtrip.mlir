// RUN: pmlc-opt %s 

// CHECK-LABEL: @layer
func @layer(%arg0: tensor<10x20xf32>) -> tensor<10x20xf32> {
  %0 = layer.box "foo" (%arg0) : tensor<10x20xf32> -> tensor<10x20xf32> {
    ^bb0(%0: tensor<10x20xf32>):
      layer.return %0 : tensor<10x20xf32>
  }
  return %0 : tensor<10x20xf32>
}
