// Test Stripe->Affine conversion
// RUN: pmlc-opt %s -convert-stripe-to-affine -split-input-file | FileCheck %s --check-prefix=AFFINE

// Test Affine->Stripe conversion
// RUN: pmlc-opt %s -convert-stripe-to-affine -split-input-file | pmlc-opt -convert-affine-to-stripe | FileCheck %s --check-prefix=STRIPE

// These tests verify Stripe/Affine round-trip dialect conversion.

// -----

func @main_parallel_for()
attributes  {stripe_attrs = {program = unit, total_macs = 27 : i64}} {
  "stripe.parallel_for"() ( {
    stripe.terminate
  }) {comments = "", name = "main", ranges = [], stripe_attrs = {main = unit}} : () -> ()
  stripe.terminate
}
// AFFINE-LABEL: func @main_parallel_for()
// AFFINE-NEXT: attributes
// AFFINE-NEXT: affine.terminator
// AFFINE-NOT: affine.terminator
// AFFINE-NOT: ^bb

// STRIPE-LABEL: func @main_parallel_for()
// STRIPE-NEXT: attributes
// STRIPE-NEXT: stripe.terminate
// STRIPE-NOT: stripe.terminate
// STRIPE-NOT: ^bb

// -----

func @affine_const()
attributes  {stripe_attrs = {program = unit, total_macs = 27 : i64}} {
  "stripe.parallel_for"() ( {
    %c0 = stripe.affine_const 0
    stripe.terminate
  }) {comments = "", name = "main", ranges = [], stripe_attrs = {main = unit}} : () -> ()
  stripe.terminate
}
// AFFINE-LABEL: func @affine_const()
// AFFINE-NEXT: attributes
// AFFINE-NEXT: constant 0 : index

// STRIPE-LABEL: func @affine_const()
// STRIPE-NEXT: attributes
// STRIPE-NEXT: stripe.affine_const 0
