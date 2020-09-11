// RUN: pmlc-opt -convert-tile-to-pxa -canonicalize -cse %s | FileCheck %s

func @scatter1d(%arg0: tensor<1xsi32>, %arg1: tensor<1x4xsi32>, %arg2: tensor<4xf32>) -> tensor<1xf32> {
  %0 = "tile.scatter"(%arg2, %arg1, %arg0) : (tensor<4xf32>, tensor<1x4xsi32>, tensor<1xsi32>) -> tensor<1xf32>
  return %0 : tensor<1xf32>
}

// -----

func @scatter3d(%arg0: tensor<3xsi32>, %arg1: tensor<1x2xsi32>, %arg2: tensor<2x4x4xf32>) -> tensor<3x4xf32> {
  %0 = "tile.scatter"(%arg2, %arg1, %arg0) : (tensor<2x4x4xf32>, tensor<1x2xsi32>, tensor<3xsi32>) -> tensor<3x4xf32>
  return %0 : tensor<3x4xf32>
}
