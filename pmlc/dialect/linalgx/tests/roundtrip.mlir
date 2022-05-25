// RUN: pmlc-opt %s | pmlc-opt | FileCheck %s

func.func @copy(%arg0: tensor<1x3x4xf32>, %arg1: tensor<1x3x2x2xf32>) -> tensor<1x3x2x2xf32> {
  %0 = linalgx.copy(%arg0, %arg1) {
    inputMap = affine_map<(i, j, k0, k1) -> (i, j, k0 * 2 + k1)>,
    outputMap = affine_map<(i, j, k0, k1) -> (i, j, k0, k1)>
  } : tensor<1x3x4xf32>, tensor<1x3x2x2xf32> -> tensor<1x3x2x2xf32>
  return %0 : tensor<1x3x2x2xf32>
}

//      CHECK: #[[map0:.*]] = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2 * 2 + d3)>
//      CHECK: #[[map1:.*]] = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>
//      CHECK: func.func @copy
//      CHECK:   linalgx.copy(%{{.*}}, %{{.*}})
// CHECK-SAME:   {inputMap = #[[map0]], outputMap = #[[map1]]}
// CHECK-SAME:   tensor<1x3x4xf32>, tensor<1x3x2x2xf32> -> tensor<1x3x2x2xf32>
