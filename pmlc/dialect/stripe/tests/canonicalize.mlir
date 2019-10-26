// RUN: pmlc-opt %s -canonicalize | FileCheck %s

!aff = type !stripe.affine
!fp32_2 = type !stripe<"tensor_ref !eltwise.fp32:2">

// CHECK-LABEL: @simplify_affines
func @simplify_affines(%arg0: !fp32_2) -> !fp32_2 {

  // This test requires the use of all of the affine canonicalization
  // patterns in order to produce code that elides the final stripe.refine
  // operation.

  // CHECK-NOT:    stripe.refine

  // Compute 5 * 2 - 8 - 2 in parts, make sure it becomes zero
  %z = stripe.affine_poly () [], 0
  %c5 = stripe.affine_poly () [], 5
  %c10 = stripe.affine_poly () [], 8 
  %t0 = stripe.affine_poly (%c5) [2], 0
  %t1 = stripe.affine_poly (%t0, %c10) [1, -1], -2

  %T = stripe.refine %arg0 (%t1, %z) : !fp32_2
  %0 = stripe.load %T : !fp32_2
  stripe.store %T, %0 : !fp32_2
  stripe.terminate
}

// CHECK-LABEL: @simplify_raw_ref
func @simplify_raw_ref(%arg0: !fp32_2) -> !fp32_2 {

  // This test makes sure the simple affines that return raw block ops are removed

  // CHECK-NOT:    stripe.affine_poly

  stripe.parallel_for ("i":100) {
  ^bb0(%i: !aff):
    %p = stripe.affine_poly (%i) [1], 0
    %T = stripe.refine %arg0 (%p, %p) : !fp32_2
    %0 = stripe.load %T : !fp32_2
    stripe.store %T, %0 : !fp32_2
    stripe.terminate
  }
  stripe.terminate
}

// CHECK-LABEL: @no_simplify_useful_refines
func @no_simplify_useful_refines(%arg0: !fp32_2) -> !fp32_2 {

  // This test validates that useful stripe.refine ops are not removed.

  // CHECK:   stripe.refine

  %c0 = stripe.affine_poly () [], 0
  %c1 = stripe.affine_poly () [], 1
  %T = stripe.refine %arg0 (%c0, %c1) : !fp32_2
  %0 = stripe.load %T : !fp32_2
  stripe.store %T, %0 : !fp32_2
  stripe.terminate
}

// CHECK-LABEL: @no_simplify_stripe_attr_refines
func @no_simplify_stripe_attr_refines(%arg0: !fp32_2) -> !fp32_2 {

  // This test validates that stripe.refine ops with attributes are not removed.

  // CHECK:   stripe.refine

  %c0 = stripe.affine_poly () [], 0
  %T = stripe.refine %arg0 (%c0, %c0) : !fp32_2 { stripe_attrs = {} }
  %0 = stripe.load %T : !fp32_2
  stripe.store %T, %0 : !fp32_2
  stripe.terminate
}

