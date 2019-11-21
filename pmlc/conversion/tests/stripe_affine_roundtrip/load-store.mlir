// Test Stripe->Affine conversion
// RUN: pmlc-opt %s -convert-stripe-to-affine -split-input-file | FileCheck %s --check-prefix=AFFINE

// Verify Stripe/Affine round-trip of stripe.load and stripe.store operations.

!fp32_1 = type !stripe<"tensor_ref !eltwise.fp32:1">
func @load_constant_addr(%arg0: !fp32_1 {stripe.layout = !stripe<"tensor !eltwise.fp32(addr[2048:1])">})
attributes {stripe_attrs = {program = unit}} {
  stripe.parallel_for () {
    %c0 = stripe.affine_poly () [], 128
    %r0 = stripe.refine %arg0 (%c0) : !fp32_1
    %l0 = stripe.load %r0 : !fp32_1
    stripe.terminate
  } {name = "main", stripe_attrs = {main = unit}} 
  stripe.terminate
}
// AFFINE-LABEL: func @load_constant_addr(
// AFFINE-SAME: %[[BASE:.*]]: memref
// AFFINE: affine.load %[[BASE]][128]

// -----

!fp32_1 = type !stripe<"tensor_ref !eltwise.fp32:1">
func @store_constant_addr(%arg0: !fp32_1 {stripe.layout = !stripe<"tensor !eltwise.fp32(addr[2048:1])">}, %arg1: !eltwise.fp32)
attributes {stripe_attrs = {program = unit}} {
  stripe.parallel_for () {
    %c0 = stripe.affine_poly () [], 128
    %r0 = stripe.refine %arg0 (%c0) : !fp32_1
    "stripe.store"(%r0, %arg1) : (!fp32_1, !eltwise.fp32) -> ()
    stripe.terminate
  } {name = "main", stripe_attrs = {main = unit}} 
  stripe.terminate
}
// AFFINE-LABEL: func @store_constant_addr(
// TODO: Remove {{.*}} for Stripe attribute in block argument when Stripe attribute is removed.
// AFFINE-SAME: %[[BASE:.*]]: memref<2048xf32> {{.*}}, %[[VAL:.*]]: f32 
// AFFINE: affine.store %[[VAL]], %[[BASE]][128]
