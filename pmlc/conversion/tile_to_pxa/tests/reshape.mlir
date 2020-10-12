// RUN: pmlc-opt -convert-tile-to-pxa -canonicalize -cse %s | FileCheck %s

func @reshaper0(%arg0: tensor<1x1x60xf32>) -> tensor<60xf32> {
  %0 = tile.reshape %arg0 : (tensor<1x1x60xf32>) -> tensor<60xf32>
  return %0 : tensor<60xf32>
}

// CHECK-LABEL: func @reshaper0
// CHECK: stdx.reshape
// CHECK: memref<60xf32>

// -----

func @reshaper1(%arg0: tensor<2x4x5xf32>) -> tensor<2x20xf32> {
  %0 = tile.reshape %arg0 : (tensor<2x4x5xf32>) -> tensor<2x20xf32>
  return %0 : tensor<2x20xf32>
}

// CHECK-LABEL: func @reshaper1
// CHECK: stdx.reshape
// CHECK: memref<2x20xf32>

// -----

func @reshaper2(%arg1: tensor<5x2x3xf32>) -> tensor<5x6xf32> {
  %0 = tile.reshape %arg1 : (tensor<5x2x3xf32>) -> tensor<5x6xf32>
  return %0 : tensor<5x6xf32>
}

// CHECK-LABEL: func @reshaper2
// CHECK: stdx.reshape
// CHECK: memref<5x6xf32>

// -----

func @squeeze(%arg0: tensor<4x2x1x3x2xf32>) -> tensor<4x2x3x2xf32> {
  %0 = tile.reshape %arg0 : (tensor<4x2x1x3x2xf32>) -> tensor<4x2x3x2xf32>
  return %0 : tensor<4x2x3x2xf32>
}

// CHECK-LABEL: func @squeeze
// CHECK: stdx.reshape
// CHECK: memref<4x2x3x2xf32>
