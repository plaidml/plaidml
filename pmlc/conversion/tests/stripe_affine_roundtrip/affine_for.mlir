// Test Stripe->Affine conversion
// RUN: pmlc-opt %s -convert-stripe-to-affine -split-input-file | FileCheck %s --check-prefix=AFFINE

// Test Affine->Stripe conversion
// RUN: pmlc-opt %s -convert-stripe-to-affine -convert-affine-to-stripe -split-input-file | FileCheck %s --check-prefix=STRIPE

// Verify Stripe/Affine round-trip conversion of stripe.parallel_for operation.

func @main_parallel_for()
attributes {stripe_attrs = {program}} {
  stripe.parallel_for () {
    stripe.terminate
  } {name = "main", stripe_attrs = {main}}
  stripe.terminate
}
// AFFINE-LABEL: func @main_parallel_for()
// AFFINE: affine.terminator
// AFFINE-NOT: affine.terminator
// AFFINE-NOT: ^bb

// STRIPE-LABEL: func @main_parallel_for()
// STRIPE: stripe.parallel_for () {
// STRIPE: stripe.terminate
// STRIPE: } {name = "main", stripe_attrs = {main}}
// STRIPE: stripe.terminate
// STRIPE-NOT: stripe.terminate
// STRIPE-NOT: ^bb
