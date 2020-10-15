// RUN: pmlc-opt -tile-materialize %s | FileCheck %s

func @relu(%arg0: tensor<10x20xf32>) -> tensor<10x20xf32> {
  %cst = "eltwise.sconst"() {value = 0.0 : f64} : () -> tensor<!eltwise.fx>
  %0 = "eltwise.cmp_lt"(%arg0, %cst) : (tensor<10x20xf32>, tensor<!eltwise.fx>) -> tensor<10x20xi1>
  %1 = "eltwise.cast"(%cst) : (tensor<!eltwise.fx>) -> tensor<f32>
  %2 = "eltwise.select"(%0, %1, %arg0) : (tensor<10x20xi1>, tensor<f32>, tensor<10x20xf32>) -> tensor<10x20xf32>
  return %2 : tensor<10x20xf32>
}

// CHECK-LABEL: func @relu
// CHECK: "eltwise.sconst"
// CHECK: "eltwise.cast"(%{{.*}}) : (tensor<!eltwise.fx>) -> tensor<f32>
// CHECK: "eltwise.cmp_lt"(%{{.*}}, %{{.*}}) : (tensor<10x20xf32>, tensor<f32>) -> tensor<10x20xi1>
