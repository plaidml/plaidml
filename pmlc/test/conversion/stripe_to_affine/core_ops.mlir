// RUN: pmlc-opt %s -convert-stripe-to-affine -split-input-file | FileCheck %s

// These tests verify stripe-to-affine dialect conversion.

// -----

// CHECK-LABEL: func @main_parallel_for()
// CHECK-NEXT: attributes
// CHECK-NEXT: affine.terminator
// CHECK-NOT: affine.terminator
// CHECK-NOT: ^bb
func @main_parallel_for()
attributes  {stripe_attrs = {program = unit, total_macs = 27 : i64}} {
  "stripe.parallel_for"() ( {
    stripe.terminate
  }) {comments = "", name = "main", ranges = [], stripe_attrs = {main = unit}} : () -> ()
  stripe.terminate
}

