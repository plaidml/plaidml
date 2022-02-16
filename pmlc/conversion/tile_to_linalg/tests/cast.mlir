// RUN: pmlc-opt -convert-tile-to-linalg %s | FileCheck %s

// CHECK-LABEL: func @cast_f32_i16
func @cast_f32_i16(%arg0: tensor<f32>) -> tensor<si16> {
  %0 = tile.cast %arg0 : (tensor<f32>) -> tensor<si16>
  // CHECK: linalg.init_tensor
  // CHECK: linalg.generic
  // CHECK: fptosi
  // CHECK: linalg.yield
  return %0 : tensor<si16>
}

// CHECK-LABEL: func @cast_f32_u16
func @cast_f32_u16(%arg0: tensor<f32>) -> tensor<ui16> {
  %0 = tile.cast %arg0 : (tensor<f32>) -> tensor<ui16>
  // CHECK: linalg.init_tensor
  // CHECK: linalg.generic
  // CHECK: fptoui
  // CHECK: linalg.yield
  return %0 : tensor<ui16>
}

// CHECK-LABEL: func @cast_i16_f32
func @cast_i16_f32(%arg0: tensor<si16>) -> tensor<f32> {
  %0 = tile.cast %arg0 : (tensor<si16>) -> tensor<f32>
  // CHECK: linalg.init_tensor
  // CHECK: linalg.generic
  // CHECK: sitofp
  // CHECK: linalg.yield
  return %0 : tensor<f32>
}

// CHECK-LABEL: func @cast_u16_f32
func @cast_u16_f32(%arg0: tensor<ui16>) -> tensor<f32> {
  %0 = tile.cast %arg0 : (tensor<ui16>) -> tensor<f32>
  // CHECK: linalg.init_tensor
  // CHECK: linalg.generic
  // CHECK: uitofp
  // CHECK: linalg.yield
  return %0 : tensor<f32>
}
