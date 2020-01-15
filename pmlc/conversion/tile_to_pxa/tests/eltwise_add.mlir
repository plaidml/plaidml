// RUN: pmlc-opt -tile-legalize-to-pxa -canonicalize -cse %s | FileCheck %s

func @eltwise_add(
  %arg0: tensor<10x20x!eltwise.fp32>,
  %arg1: tensor<10x20x!eltwise.fp32>
) -> tensor<10x20x!eltwise.fp32> {
  %0 = "eltwise.add"(%arg1, %arg0) : (
    tensor<10x20x!eltwise.fp32>,
    tensor<10x20x!eltwise.fp32>
  ) -> tensor<10x20x!eltwise.fp32>
  return %0 : tensor<10x20x!eltwise.fp32>
}

// CHECK-LABEL: func @eltwise_add
// CHECK: pxa.parallel_for
// CHECK: affine.load
// CHECK: affine.load
// CHECK: addf
// CHECK: affine.store
