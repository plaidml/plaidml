// Test Stripe->Affine conversion
// RUN: pmlc-opt %s -convert-stripe-to-affine -split-input-file | FileCheck %s --check-prefix=AFFINE

// Verify Stripe/Affine round-trip of stripe.parallel_for.

!fp32_1 = type !stripe<"tensor_ref !eltwise.fp32:1">
!fp32 = type tensor<!eltwise.fp32>
func @simple_1D_for(%arg0: !fp32_1 {stripe.layout = !stripe<"tensor !eltwise.fp32([2048:1])">}, %arg1: !fp32_1 {stripe.layout = !stripe<"tensor !eltwise.fp32([2048:1])">}) {
  stripe.parallel_for ("i":2048) {
  ^bb0(%i: !stripe.affine):
    %p0 = stripe.affine_poly (%i) [1], 0
    %r0 = stripe.refine %arg0 (%p0) : !fp32_1
    %l0 = stripe.load %r0 : !fp32_1
    %r1 = stripe.refine %arg1 (%p0) : !fp32_1
    "stripe.store"(%r1, %l0) : (!fp32_1, !fp32) -> ()
    stripe.terminate
  }
  stripe.terminate
}
// AFFINE-LABEL: func @simple_1D_for(
// AFFINE-SAME: %[[IN:.*]]: memref<2048xf32> {{.*}}, %[[OUT:.*]]: memref<2048xf32>
// AFFINE: affine.for %[[i:.*]] = 0 to 2048
// AFFINE: %[[VAL:.*]] = affine.load %[[IN]][%[[i]]]
// AFFINE: affine.store %[[VAL]], %[[OUT]][%[[i]]]

// -----

!fp32_3 = type !stripe<"tensor_ref !eltwise.fp32:3">
!fp32 = type tensor<!eltwise.fp32>
func @simple_multi_dim_for(%arg0: !fp32_3 {stripe.layout = !stripe<"tensor !eltwise.fp32([2:32], [4:8], [8:1])">}, %arg1: !fp32_3 {stripe.layout = !stripe<"tensor !eltwise.fp32([2:32], [4:8], [8:1])">}) {
  stripe.parallel_for ("i":2, "j":4, "k":8) {
  ^bb0(%i: !stripe.affine, %j: !stripe.affine, %k: !stripe.affine):
    %pi = stripe.affine_poly (%i) [1], 0
    %pj = stripe.affine_poly (%j) [1], 0
    %pk = stripe.affine_poly (%k) [1], 0
    %r0 = stripe.refine %arg0 (%pi, %pj, %pk) : !fp32_3
    %l0 = stripe.load %r0 : !fp32_3
    %r1 = stripe.refine %arg1 (%pi, %pj, %pk) : !fp32_3
    "stripe.store"(%r1, %l0) : (!fp32_3, !fp32) -> ()
    stripe.terminate
  }
  stripe.terminate
}
// AFFINE-LABEL: func @simple_multi_dim_for(
// AFFINE-SAME: %[[IN:.*]]: memref<2x4x8xf32> {{.*}}, %[[OUT:.*]]: memref<2x4x8xf32>
// AFFINE: affine.for %[[i:.*]] = 0 to 2
// AFFINE:   affine.for %[[j:.*]] = 0 to 4
// AFFINE:     affine.for %[[k:.*]] = 0 to 8
// AFFINE:       %[[VAL:.*]] = affine.load %[[IN]][%[[i]], %[[j]], %[[k]]]
// AFFINE:       affine.store %[[VAL]], %[[OUT]][%[[i]], %[[j]], %[[k]]]
