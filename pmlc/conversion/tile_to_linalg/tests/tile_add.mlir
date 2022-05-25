// RUN: pmlc-opt -convert-tile-to-linalg %s | FileCheck %s

func.func @main(%arg0: tensor<4xsi32> {stdx.const}, 
           %arg1: tensor<4xsi32> {stdx.const}) -> tensor<4xsi32> {
  %0 = tile.add %arg0, %arg1 : (tensor<4xsi32>, tensor<4xsi32>) -> tensor<4xsi32>
  return %0 : tensor<4xsi32>
}

// CHECK-LABEL: @main
