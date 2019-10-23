// RUN: pmlc-opt %s -canonicalize | FileCheck %s

!fp32_2 = type !stripe<"tensor_ref !eltwise.fp32:2">

// CHECK-LABEL: @simplify_affines
func @simplify_affines(%arg0: !fp32_2) -> !fp32_2 {

  // This test requires the use of all of the affine canonicalization
  // patterns in order to produce code that elides the final stripe.refine
  // operation.

  // CHECK-NOT:    stripe.refine

  %c5 = stripe.affine_const 5
  %m0 = stripe.affine_mul %c5, 0

  %m5 = stripe.affine_mul %c5, 1
  %s5 = stripe.affine_add (%m5)

  %cNeg1 = stripe.affine_const -1
  %mNeg5 = stripe.affine_mul %cNeg1, 5

  %empty = stripe.affine_add ()
  %z = stripe.affine_add (%s5, %mNeg5, %empty)

  %T = stripe.refine %arg0 (%m0, %z) : !fp32_2
  %0 = stripe.load %T : !fp32_2
  stripe.store %T, %0 : !fp32_2
  stripe.terminate
}

// CHECK-LABEL: @no_simplify_useful_refines
func @no_simplify_useful_refines(%arg0: !fp32_2) -> !fp32_2 {

  // This test validates that useful stripe.refine ops are not removed.

  // CHECK:   stripe.refine

  %c0 = stripe.affine_const 0
  %c1 = stripe.affine_const 1
  %T = stripe.refine %arg0 (%c0, %c1) : !fp32_2
  %0 = stripe.load %T : !fp32_2
  stripe.store %T, %0 : !fp32_2
  stripe.terminate
}
