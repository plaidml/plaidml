// RUN: pmlc-opt %s -convert-stripe-to-affine | FileCheck %s

// These tests verify stripe-to-affine dialect conversion.

// WIP
module {
  func @parallel_for()
  attributes  {stripe_attrs = {program = unit, total_macs = 27 : i64}} {
    "stripe.parallel_for"() ( {
      stripe.terminate
    }) {comments = "", name = "main", ranges = [], stripe_attrs = {main = unit}} : () -> ()
    stripe.terminate
  }
}

