// RUN: pmlc-opt -pass-pipeline='tile-constant-types{floatx=f64 intx=ui64}' -canonicalize %s | FileCheck %s

module {
  func @higher_precision_constants(%arg0: tensor<3x3xf32>) -> tensor<3x3xf32> {
    %c1 = "eltwise.sconst"() {value = 1 : i64} : () -> tensor<si64>
    %cst = "eltwise.sconst"() {value = 2.000000e+00 : f64} : () -> tensor<f32>
    %0 = "eltwise.add"(%arg0, %c1) : (tensor<3x3xf32>, tensor<si64>) -> tensor<3x3xf32>
    %1 = "eltwise.add"(%0, %cst) : (tensor<3x3xf32>, tensor<f32>) -> tensor<3x3xf32>
    return %1 : tensor<3x3xf32>
  }
}

// CHECK-LABEL: func @higher_precision_constants
// CHECK: eltwise.sconst
// CHECK-SAME: {value = 1 : i64} : () -> tensor<ui64>
// CHECK: eltwise.sconst
// CHECK-SAME: {value = 2.000000e+00 : f64} : () -> tensor<f64>
// CHECK: return
// CHECK-SAME: tensor<3x3xf64>
