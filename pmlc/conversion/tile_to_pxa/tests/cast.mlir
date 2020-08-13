// RUN: pmlc-opt -convert-tile-to-pxa -canonicalize -cse -split-input-file %s | FileCheck %s

// CHECK-LABEL: func @cast_sconst
func @cast_sconst() -> tensor<f32> {
  %c3 = "eltwise.sconst"() {value = 3 : i64} : () -> tensor<si32>
  // CHECK: alloc
  // CHECK: sitofp
  // CHECK: pxa.reduce assign
  %1 = "eltwise.cast"(%c3) : (tensor<si32>) -> tensor<f32>
  return %1 : tensor<f32>
}

// -----

// CHECK-LABEL: func @cast_f32_i16
func @cast_f32_i16(%arg0: tensor<f32>) -> tensor<si16> {
  %0 = "eltwise.cast"(%arg0) : (tensor<f32>) -> tensor<si16>
  // CHECK: alloc
  // CHECK: pxa.load
  // CHECK: fptosi
  // CHECK: pxa.reduce assign
  return %0 : tensor<si16>
}

// -----

// CHECK-LABEL: func @cast_f32_u16
func @cast_f32_u16(%arg0: tensor<f32>) -> tensor<ui16> {
  %0 = "eltwise.cast"(%arg0) : (tensor<f32>) -> tensor<ui16>
  // CHECK: alloc
  // CHECK: pxa.load
  // CHECK: stdx.fptoui
  // CHECK: pxa.reduce assign
  return %0 : tensor<ui16>
}

// -----

// CHECK-LABEL: func @cast_i16_f32
func @cast_i16_f32(%arg0: tensor<si16>) -> tensor<f32> {
  %0 = "eltwise.cast"(%arg0) : (tensor<si16>) -> tensor<f32>
  // CHECK: alloc
  // CHECK: pxa.load
  // CHECK: sitofp
  // CHECK: pxa.reduce assign
  return %0 : tensor<f32>
}

// -----

// CHECK-LABEL: func @cast_u16_f32
func @cast_u16_f32(%arg0: tensor<ui16>) -> tensor<f32> {
  %0 = "eltwise.cast"(%arg0) : (tensor<ui16>) -> tensor<f32>
  // CHECK: alloc
  // CHECK: pxa.load
  // CHECK: stdx.uitofp
  // CHECK: pxa.reduce assign
  return %0 : tensor<f32>
}
