// RUN: pmlc-opt -tile-split-main="main-function=main" %s | FileCheck %s

// CHECK-LABEL: @main
func @main(%arg0: tensor<16x16xf32>) -> tensor<16x16xf32> {
  return %arg0 : tensor<16x16xf32>
}

