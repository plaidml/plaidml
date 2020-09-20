// RUN: pmlc-opt -canonicalize %s | FileCheck %s

func @basic() -> tensor<3xf32> {
  %0 = constant dense<[0.0, 0.0, 0.0]> : tensor<3xf32>
  %1 = constant dense<[1.0, 1.0, 1.0]> : tensor<3xf32>
  %2 = "eltwise.add"(%0, %1) : (tensor<3xf32>, tensor<3xf32>) -> tensor<3xf32>
  return %2 : tensor<3xf32>
}
