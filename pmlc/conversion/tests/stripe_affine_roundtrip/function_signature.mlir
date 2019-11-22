// Test Stripe->Affine conversion
// RUN: pmlc-opt %s -convert-stripe-to-affine -split-input-file | FileCheck %s --check-prefix=AFFINE

!fp32_4 = type !stripe<"tensor_ref !eltwise.fp32:2">
func @tensor_func(%arg0: !fp32_4 {stripe.layout = !stripe<"tensor !eltwise.fp32(addr[10:20], addr[20:1])">})
attributes {stripe_attrs = {program = unit}} {
  stripe.terminate
}
// TODO: Remove 'tensor' attribue.
// AFFINE: func @tensor_func(%arg0: memref<10x20xf32>

// -----

func @float_func(%arg0: !eltwise.fp32, %arg1: !eltwise.fp64)
attributes {stripe_attrs = {program = unit}} {
  stripe.terminate
}
// AFFINE: func @float_func(%arg0: f32, %arg1: f64

// -----

func @int_func(%arg0: !eltwise.i8, %arg1: !eltwise.i64)
attributes {stripe_attrs = {program = unit}} {
  stripe.terminate
}
// AFFINE: func @int_func(%arg0: i8, %arg1: i64
