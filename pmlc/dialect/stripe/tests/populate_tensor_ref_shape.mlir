// RUN: pmlc-opt %s -test-populate-tensor-ref-shape -split-input-file | FileCheck %s

// Verify PopulateTensorRefShape analysis.

module {
  func @eltwise_add(%arg0: !stripe<"tensor_ref !eltwise.fp32:2"> {stripe.layout = !stripe<"tensor !eltwise.fp32(addr[10:20], addr[20:1])">, stripe.name = "_X0"})
  attributes  {stripe_attrs = {program = unit}} {
    stripe.parallel_for () {
      %c0 = stripe.affine_poly () [], 0
      %0 = stripe.refine %arg0(%c0, %c0) : !stripe<"tensor_ref !eltwise.fp32:2">
      stripe.terminate
    } {name = "main", stripe_attrs = {main = unit}}
    stripe.terminate
  }
}
// AFFINE-LABEL: func @constant_affine_poly()
// AFFINE: constant 0 : index

// STRIPE-LABEL: func @constant_affine_poly()
// STRIPE: stripe.affine_poly () [], 0
