// RUN: pmlc-opt -convert-tile-to-pxa -canonicalize -cse %s | FileCheck %s

module {
  func @reshaper(%arg0: tensor<1x1x60xf32>) -> tensor<60xf32> {
    %c60 = tile.constant 60
    %0 = "tile.reshape"(%arg0, %c60) : (tensor<1x1x60xf32>, index) -> tensor<60xf32>
    return %0 : tensor<60xf32>
  }
}

// CHECK-LABEL: func @reshaper
// CHECK: stdx.reshape
// CHECK: memref<60xf32>

// -----

module {
  func @reshaper(%arg0: tensor<2x4x5xf32>) -> tensor<2x20xf32> {
    %c2 = tile.constant 2
    %c20 = tile.constant 20
    %0 = "tile.reshape"(%arg0, %c2, %c20) : (tensor<2x4x5xf32>, index, index) -> tensor<2x20xf32>
    return %0 : tensor<2x20xf32>
  }
}

// CHECK-LABEL: func @reshaper
// CHECK: stdx.reshape
// CHECK: memref<2x20xf32>

// -----

module {
  func @reshaper(%arg0: tensor<5x6xf32>, %arg1: tensor<5x2x3xf32>) -> tensor<5x6xf32> {
    %c5 = tile.constant 5
    %c6 = tile.constant 6
    %0 = "tile.reshape"(%arg1, %c5, %c6) : (tensor<5x2x3xf32>, index, index) -> tensor<5x6xf32>
    return %0 : tensor<5x6xf32>
  }
}

// CHECK-LABEL: func @reshaper
// CHECK: stdx.reshape
// CHECK: memref<5x6xf32>

// -----

module {
  func @squeeze(%arg0: tensor<4x2x1x3x2xf32>) -> tensor<4x2x3x2xf32> {
    %c4 = tile.constant 4
    %c3 = tile.constant 3
    %c2 = tile.constant 2
    %0 = "tile.reshape"(%arg0, %c4, %c2, %c3, %c2) : (tensor<4x2x1x3x2xf32>, index, index, index, index) -> tensor<4x2x3x2xf32>
    return %0 : tensor<4x2x3x2xf32>
  }
}

// CHECK-LABEL: func @squeeze
// CHECK: stdx.reshape
// CHECK: memref<4x2x3x2xf32>
