// RUN: pmlc-opt -convert-tile-to-pxa -canonicalize -cse %s | FileCheck %s

func @relu(%arg0: tensor<10x20xf32>) -> tensor<10x20xf32> {
  %0 = tile.constant(0.0 : f32) : tensor<f32>
  %1 = tile.cmp_lt %arg0, %0 : (tensor<10x20xf32>, tensor<f32>) -> tensor<10x20xi1>
  %2 = tile.select %1, %0, %arg0 : (tensor<10x20xi1>, tensor<f32>, tensor<10x20xf32>) -> tensor<10x20xf32>
  return %2 : tensor<10x20xf32>
}

// CHECK-LABEL: func @relu
// CHECK: alloc
// CHECK: affine.parallel
// CHECK: pxa.load
// CHECK: cmpf "olt"
// CHECK: pxa.reduce assign
// CHECK: affine.parallel
// CHECK: pxa.load
// CHECK: pxa.load
// CHECK: select
// CHECK: pxa.reduce assign
