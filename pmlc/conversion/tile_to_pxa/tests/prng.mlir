// RUN: pmlc-opt -convert-tile-to-pxa -canonicalize -cse %s | FileCheck %s

module {
  func @prng(%arg0: tensor<3x2048xui32>) -> (tensor<2x3x4x5xf32>, tensor<3x2048xui32>) {
    %result, %new_state = "tile.prng"(%arg0) : (tensor<3x2048xui32>) -> (tensor<2x3x4x5xf32>, tensor<3x2048xui32>)
    return %result, %new_state : tensor<2x3x4x5xf32>, tensor<3x2048xui32>
  }
}

// CHECK-LABEL: func @prng
// CHECK "pxa.prng"(%{{.*}}, %{{.*}}, %{{.*}})
