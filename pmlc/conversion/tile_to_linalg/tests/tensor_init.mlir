// RUN: pmlc-opt -convert-tile-to-linalg %s | FileCheck %s

func @contract_tensor_init(%arg0: tensor<3x3xf32>, %arg1: tensor<3xf32>, %arg2: tensor<3xf32>) -> tensor<3xf32> {
  %0 = tile.contract add, mul, %arg2, %arg0, %arg1 {lowerBounds = affine_map<() -> (0, 0)>, sink = affine_map<(d0, d1) -> (d0)>, srcs = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d1)>], upperBounds = affine_map<() -> (2, 2)>} : tensor<3xf32>, tensor<3x3xf32>, tensor<3xf32> -> tensor<3xf32>
  return %0 : tensor<3xf32>
}

// CHECK-LABEL: func @contract_tensor_init
// CHECK: linalg.init_tensor
// CHECK: linalg.generic
// CHECK:   linalg.yield
// CHECK:   linalg.generic
// CHECK:     mulf
// CHECK:     addf
// CHECK:     linalg.yield
// CHECK: return
