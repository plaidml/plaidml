// RUN: pmlc-opt -canonicalize %s | FileCheck %s

func @basic() -> tensor<2x2xf32> {
  %cst = "eltwise.sconst"() {value = 1.0 : f64} : () -> f32
  %0 = tile.constant_tensor @weights : tensor<2x2xf32>
  %1 = "eltwise.add"(%cst, %0) : (f32, tensor<2x2xf32>) -> tensor<2x2xf32>
  return %1 : tensor<2x2xf32>
}
