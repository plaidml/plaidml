// RUN: pmlc-opt -canonicalize %s | FileCheck %s

func @basic() -> tensor<2x2xf32> {
  %cst = eltwise.constant 1.0 : f32
  %0 = constant dense<[[1.0, 2.0], [3.0, 4.0]]> : tensor<2x2xf32>
  %1 = "eltwise.add"(%cst, %0) : (f32, tensor<2x2xf32>) -> tensor<2x2xf32>
  return %1 : tensor<2x2xf32>
}
