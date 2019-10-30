// Test Stripe->Affine conversion
// RUN: pmlc-opt %s -convert-stripe-to-affine -split-input-file | FileCheck %s --check-prefix=AFFINE

// Test Affine->Stripe conversion
// RUN: pmlc-opt %s -convert-stripe-to-affine -convert-affine-to-stripe -split-input-file | FileCheck %s --check-prefix=STRIPE

// Verify Stripe/Affine round-trip of affine.poly operation.

func @constant_affine_poly()
attributes {stripe_attrs = {program = unit}} {
  stripe.parallel_for () {
    %c0 = stripe.affine_poly () [], 0
    stripe.terminate
  } {name = "main", stripe_attrs = {main = unit}} 
  stripe.terminate
}
// AFFINE-LABEL: func @constant_affine_poly()
// AFFINE: constant 0 : index

// STRIPE-LABEL: func @constant_affine_poly()
// STRIPE: stripe.affine_poly () [], 0
