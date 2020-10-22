// RUN: pmlc-opt -convert-tile-to-pxa -canonicalize -cse -split-input-file %s | FileCheck %s

// CHECK-LABEL: func @cast_f32_i16
func @cast_f32_i16(%arg0: tensor<f32>) -> tensor<si16> {
  %0 = tile.cast %arg0 : (tensor<f32>) -> tensor<si16>
  // CHECK: pxa.load
  // CHECK: fptosi
  // CHECK: pxa.reduce assign
  return %0 : tensor<si16>
}

// -----

// CHECK-LABEL: func @cast_f32_u16
func @cast_f32_u16(%arg0: tensor<f32>) -> tensor<ui16> {
  %0 = tile.cast %arg0 : (tensor<f32>) -> tensor<ui16>
  // CHECK: pxa.load
  // CHECK: fptoui
  // CHECK: pxa.reduce assign
  return %0 : tensor<ui16>
}

// -----

// CHECK-LABEL: func @cast_i16_f32
func @cast_i16_f32(%arg0: tensor<si16>) -> tensor<f32> {
  %0 = tile.cast %arg0 : (tensor<si16>) -> tensor<f32>
  // CHECK: pxa.load
  // CHECK: sitofp
  // CHECK: pxa.reduce assign
  return %0 : tensor<f32>
}

// -----

// CHECK-LABEL: func @cast_u16_f32
func @cast_u16_f32(%arg0: tensor<ui16>) -> tensor<f32> {
  %0 = tile.cast %arg0 : (tensor<ui16>) -> tensor<f32>
  // CHECK: pxa.load
  // CHECK: uitofp
  // CHECK: pxa.reduce assign
  return %0 : tensor<f32>
}
