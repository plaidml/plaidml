// RUN: pmlc-opt -convert-tile-to-pxa -canonicalize -cse -split-input-file %s | FileCheck %s

// CHECK-LABEL: func @cast_f32_i16
func @cast_f32_i16(%arg0: tensor<f32>) -> tensor<si16> {
  %0 = "eltwise.cast"(%arg0) : (tensor<f32>) -> tensor<si16>
  // CHECK: stdx.fptosi
  return %0 : tensor<si16>
}

// -----

// CHECK-LABEL: func @cast_f32_u16
func @cast_f32_u16(%arg0: tensor<f32>) -> tensor<ui16> {
  %0 = "eltwise.cast"(%arg0) : (tensor<f32>) -> tensor<ui16>
  // CHECK: stdx.fptoui
  return %0 : tensor<ui16>
}
