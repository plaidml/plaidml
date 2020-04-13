// RUN: pmlc-opt -convert-tile-to-pxa -canonicalize -cse %s | FileCheck %s

func @eltwise_add(
  %arg0: tensor<10x20xf32>,
  %arg1: tensor<10x20xf32>
) -> tensor<10x20xf32> {
  %0 = "eltwise.add"(%arg1, %arg0) : (
    tensor<10x20xf32>,
    tensor<10x20xf32>
  ) -> tensor<10x20xf32>
  return %0 : tensor<10x20xf32>
}

// CHECK-LABEL: func @eltwise_add
// CHECK: affine.parallel
// CHECK: affine.load
// CHECK: affine.load
// CHECK: addf
// CHECK: affine.store

func @eltwise_add_f32_index(%arg0: tensor<4x1xf32>) -> (tensor<4x1xf32>) {
  %c7 = tile.constant 7
  %1 = "eltwise.add"(%arg0, %c7) : (tensor<4x1xf32>, index) -> tensor<4x1xf32>
  return %1 : tensor<4x1xf32>
}
